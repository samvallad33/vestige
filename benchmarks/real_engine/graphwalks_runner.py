"""Pinned GraphWalks inputs and execution of the official README scorer.

No graph traversal, reachability calculation, or answer synthesis lives here.
Source graph text is separated from the official operation by literal format
markers only. Gold node sets are available solely to the post-merge scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import hashlib
import json
from pathlib import Path
import re
import random
from typing import Any, Mapping, Sequence
import uuid

REVISION = "be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7"
README_SHA256 = "c217a1631115f8188a804b16e9fc5bb4f8cb4dde92a49bc47ea8de504d3d4ca9"
SHORT_FILE = "graphwalks_128k_and_shorter.parquet"
SHORT_SHA256 = "54036036c91d8e04bb2a5fcd9e36f8e2a852cacece5dfc2b1ee40e3a6182b516"
SHORT_BYTES = 22128284
SHORT_ITEMS = 750
LONG_FILE = "graphwalks_256k_to_1mil.parquet"
LONG_SHA256 = "537879431c72a42e3b500f80efc3047e7facb90390b6063d33679b4320985911"
LONG_BYTES = 296903814
LONG_ITEMS = 400
# Deterministic raw-row manifest generated from the pinned long parquet. This
# seals both derived row hashes and gold metadata against cache-only edits.
LONG_PREPARED_MANIFEST_SHA256 = "9593d572988313c5d0bfa0a43ff7a865085f0f799e132b1d3dd6a749877d186b"
GRAPH_HEADER = "Here is the graph to operate on:\nThe graph has the following edges:\n"
OPERATION_HEADER = "\nOperation:\n"
CONTEXT_LIMIT = 131072
OUTPUT_BUDGET = 2048


def _root(root: str | Path) -> Path:
    root = Path(root)
    return root if (root / SHORT_FILE).exists() else root / "graphwalks"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class GraphWalkItem:
    id: str
    full_prompt: str
    question: str
    history: list[str]
    category: str
    scorer_payload: dict[str, Any] = field(repr=False)

    @property
    def group_id(self):
        return self.id

    def solver_record(self):
        return {"id": self.id, "group_id": self.id, "full_prompt": self.full_prompt,
                "question": self.question, "history": self.history}


@dataclass
class GraphWalkRef:
    """Lazy raw row: avoid keeping the long partition's huge graphs in RAM."""
    id: str
    row_path: Path
    row_sha256: str
    category: str
    scorer_payload: dict[str, Any] = field(repr=False)

    def _row(self):
        raw = self.row_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.row_sha256:
            raise ValueError("prepared GraphWalks raw row hash mismatch")
        return json.loads(raw)

    @property
    def full_prompt(self):
        return self._row()["prompt"]

    @property
    def question(self):
        return separate_graph(self.full_prompt)[0]

    @property
    def history(self):
        return separate_graph(self.full_prompt)[1]

    @property
    def group_id(self):
        return self.id

    def solver_record(self):
        return {"id": self.id, "group_id": self.id, "full_prompt": self.full_prompt,
                "question": self.question, "history": self.history}


def separate_graph(prompt: str) -> tuple[str, list[str]]:
    """Separate raw edge records by the official literal prompt delimiters."""
    if prompt.count(GRAPH_HEADER) != 1:
        raise ValueError("GraphWalks official graph header missing or duplicated")
    prefix, _, rest = prompt.partition(GRAPH_HEADER)
    graph, separator, operation = rest.rpartition(OPERATION_HEADER)
    if not separator:
        raise ValueError("GraphWalks official operation delimiter missing")
    history = [line for line in graph.splitlines() if line]
    if not history or any(line.count(" -> ") != 1 for line in history):
        raise ValueError("GraphWalks source graph is not the official raw directed edge list")
    # All worked examples, operation wording, and final-answer instructions
    # remain exactly official text. Only actual graph placement changes.
    question = prefix + GRAPH_HEADER + OPERATION_HEADER + operation
    return question, history


def verify_sources(root: str | Path, short_only=False) -> dict[str, Any]:
    root = _root(root)
    pins = [(SHORT_FILE, SHORT_SHA256), ("README.md", README_SHA256)]
    if not short_only:
        pins.append((LONG_FILE, LONG_SHA256))
    for name, expected in pins:
        if _sha(root / name) != expected:
            raise ValueError(f"GraphWalks pinned source hash mismatch: {name}")
    if (root / SHORT_FILE).stat().st_size != SHORT_BYTES:
        raise ValueError("GraphWalks short-data byte count mismatch")
    if not short_only and (root / LONG_FILE).stat().st_size != LONG_BYTES:
        raise ValueError("GraphWalks long-data byte count mismatch")
    return {"revision": REVISION, "dataset_sha256": SHORT_SHA256,
            "dataset_hashes": {name: sha for name, sha in pins if name.endswith(".parquet")},
            "scorer_source_sha256": README_SHA256, "file": SHORT_FILE,
            "scope": "short-partition-only-unit-test" if short_only else "both official partitions,1150items"}


def load_graphwalks(root: str | Path, short_only=False) -> list[GraphWalkItem | GraphWalkRef]:
    import pyarrow.parquet as pq
    root = _root(root)
    verify_sources(root, short_only=short_only)
    records = pq.read_table(root / SHORT_FILE).to_pylist()
    if len(records) != SHORT_ITEMS:
        raise ValueError(f"expected 750 official short GraphWalks rows, got {len(records)}")
    output = []
    for index, row in enumerate(records):
        prompt = row["prompt"]
        question, history = separate_graph(prompt)
        output.append(GraphWalkItem(
            id=f"graphwalks-short:{index:04d}", full_prompt=prompt,
            question=question, history=history, category=row["problem_type"],
            scorer_payload={"answer_nodes": list(row["answer_nodes"]), "row": index,
                            "prompt_chars": row["prompt_chars"], "date_added": row["date_added"]},
        ))
    if not short_only:
        # Preserve all400 long rows. Streaming preparation does no graph
        # computation and prevents a multi-GB in-memory prompt/edge duplicate.
        from .runtime import atomic_write, canonical
        directory = root / "prepared-long"
        manifest_path = directory / "rows.json"
        if manifest_path.exists():
            if _sha(manifest_path) != LONG_PREPARED_MANIFEST_SHA256:
                raise ValueError("GraphWalks prepared long manifest differs from pinned raw-source conversion")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("source_sha256") != LONG_SHA256 or len(manifest.get("rows", [])) != LONG_ITEMS:
                raise ValueError("GraphWalks prepared long-partition identity mismatch")
        else:
            manifest = {"source_sha256": LONG_SHA256, "rows": []}
            parquet = pq.ParquetFile(root / LONG_FILE)
            if parquet.metadata.num_rows != LONG_ITEMS:
                raise ValueError("expected400 official long GraphWalks rows")
            index = 0
            for batch in parquet.iter_batches(batch_size=1):
                for row in batch.to_pylist():
                    # Format validation only; preserve source text and labels
                    # in a sealed raw-source file, then separate them on access.
                    separate_graph(row["prompt"])
                    raw = canonical(row) + "\n"
                    path = directory / f"{index:04d}.json"
                    atomic_write(path, raw)
                    manifest["rows"].append({"row": index, "file": path.name,
                        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                        "category": row["problem_type"], "answer_nodes": row["answer_nodes"],
                        "prompt_chars": row["prompt_chars"], "date_added": row["date_added"]})
                    index += 1
            atomic_write(manifest_path, canonical(manifest) + "\n")
            if _sha(manifest_path) != LONG_PREPARED_MANIFEST_SHA256:
                raise ValueError("GraphWalks regenerated long manifest differs from pinned source conversion")
        output.extend(GraphWalkRef(id=f"graphwalks-long:{row['row']:04d}",
                      row_path=directory / row["file"], row_sha256=row["sha256"],
                      category=row["category"], scorer_payload={"answer_nodes": row["answer_nodes"],
                      "row": row["row"], "prompt_chars": row["prompt_chars"], "date_added": row["date_added"]})
                      for row in manifest["rows"])
        if len(output) != SHORT_ITEMS + LONG_ITEMS:
            raise ValueError("full GraphWalks denominator must remain1150")
    return output


def official_scorer(root: str | Path):
    """Execute the exact pinned publisher extraction/grading code, not a port."""
    root = _root(root)
    if _sha(root / "README.md") != README_SHA256:
        raise ValueError("GraphWalks official scorer README hash mismatch")
    text = (root / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    if len(blocks) != 2 or not blocks[0].startswith("def get_list(self, response:"):
        raise ValueError("GraphWalks official scorer block layout changed")
    # The publisher supplies get_list as a method and calls it bound in the
    # grading snippet. Binding its unused self argument changes no scorer code.
    safe_builtins = {"str": str, "bool": bool, "tuple": tuple, "list": list,
                     "set": set, "len": len}
    namespace = {"re": re, "__builtins__": safe_builtins}
    exec(compile(blocks[0], "official_graphwalks_README_extraction", "exec"), namespace)
    extract = namespace["get_list"]
    grading = compile(blocks[1], "official_graphwalks_README_grading", "exec")

    def score(response: str, answer_nodes: Sequence[str]) -> dict[str, Any]:
        env = {"get_list": lambda value: extract(None, value), "response": response,
               "answer_nodes": answer_nodes, "__builtins__": safe_builtins}
        exec(grading, env)
        return {"precision": env["precision"], "recall": env["recall"], "f1": env["f1"],
                "failed_to_parse": env["failed_to_parse"], "sampled_nodes": env["sampled_list"]}

    return score


def score_graphwalks(root: str | Path, items: Sequence[GraphWalkItem],
                    predictions: Mapping[str, str]) -> list[dict[str, Any]]:
    expected = [item.id for item in items]
    if len(expected) != len(set(expected)) or set(predictions) != set(expected):
        raise ValueError("GraphWalks official scoring requires unique exact merged coverage")
    score = official_scorer(root)
    return [{"item_id": item.id, **score(predictions[item.id], item.scorer_payload["answer_nodes"])}
            for item in items]


def graph_record_plan(raw_edges: Sequence[str]) -> dict[str, Any]:
    """Source-only incidence representation; no operation/gold accepted here.

    Endpoint identities preserve first appearance in the released raw edge
    list. Each edge is evidence about its two endpoints. This is an incidence
    representation of source facts, not a computed traversal result. Edge
    direction stays in the untouched raw edge content for the model to read.
    """
    nodes, seen, edges = [], set(), []
    for raw in raw_edges:
        if raw.count(" -> ") != 1:
            raise ValueError("source edge must have exactly one official direction delimiter")
        source, target = raw.split(" -> ")
        if not source or not target or source.strip() != source or target.strip() != target:
            raise ValueError("source node identifiers must be preserved exactly")
        for node in (source, target):
            if node not in seen:
                seen.add(node)
                nodes.append(node)
        edges.append({"raw": raw, "source": source, "target": target})
    plan = {"representation": "raw-edge-incidence-evidence-of-v1", "nodes": nodes, "edges": edges}
    from .runtime import canonical
    plan["conversion_sha256"] = hashlib.sha256(canonical(plan).encode()).hexdigest()
    return plan


def admit_graph(engine, raw_edges: Sequence[str]) -> dict[str, Any]:
    """Every record/link is admitted by the real binary's normal user path."""
    plan = graph_record_plan(raw_edges)
    node_ids, edge_ids = {}, []
    for raw_id in plan["nodes"]:
        result = engine.call("smart_ingest", {
            "content": raw_id, "node_type": "note", "tags": ["graph-node", raw_id],
            "source": "official-source-graph",
        })
        node_id = result.get("nodeId") or result.get("echoNodeId")
        if not node_id:
            raise RuntimeError("normal engine gate did not admit graph endpoint: " + repr(result))
        node_ids[raw_id] = node_id
    for edge in plan["edges"]:
        targets = list(dict.fromkeys((node_ids[edge["source"]], node_ids[edge["target"]])))
        result = engine.call("smart_ingest", {
            "content": edge["raw"], "node_type": "note", "tags": ["history"],
            "source": "official-source-graph",
            "links": [{"kind": "evidence_of", "to": target} for target in targets],
        })
        node_id = result.get("nodeId") or result.get("echoNodeId")
        if not node_id or result.get("linkError"):
            raise RuntimeError("normal engine gate did not admit graph incidence links: " + repr(result))
        admitted = result.get("links")
        if not isinstance(admitted, list) or len(admitted) < len(targets):
            raise RuntimeError("native graph links lack their admission receipts")
        actual = {(link.get("kind"), link.get("source"), link.get("target"))
                  for link in admitted if link.get("receiptId")}
        if not {("evidence_of", node_id, target) for target in targets}.issubset(actual):
            raise RuntimeError("native graph admission omitted a source-only incidence link")
        edge_ids.append(node_id)
    return {
        "conversion_sha256": plan["conversion_sha256"], "endpoint_ids": node_ids,
        "edge_ids": edge_ids, "endpoint_count": len(node_ids), "edge_count": len(edge_ids),
        "native_writes": len(node_ids) + len(edge_ids),
        "claim_limit": "Native graph records are an evidence-of incidence representation. "
                       "Raw edge strings preserve source directions; the model computes the operation. "
                       "Native GhostLink map is limited to depth3 and200nodes per call. "
                       "No host graph traversal, answer-derived links, or answer injection is used.",
    }


async def solve_graphwalks(campaign, item: GraphWalkItem, arm: str, trial: int, worker: int):
    from .engine import HookedRealEngine
    from .prompts import system_prompt
    from .provider import MODEL
    from .retrieval import BM25Memory, BGEMemory
    from .run import SEARCH_TOOL
    from .runtime import atomic_write, canonical, utc_now

    root = campaign.runs / "graphwalks" / arm / f"worker-{worker}" / f"{item.id.rsplit(':',1)[-1]}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    plan = graph_record_plan(item.history)
    meta = {"benchmark": "graphwalks", "arm": arm, "item_id": item.id,
            "trial": trial, "seed": trial, "shard": worker, "worker_id": worker,
            "binary_sha256": campaign.manifest["binary_sha256"], "started_at": utc_now(),
            "dataset_sha256": LONG_SHA256 if item.id.startswith("graphwalks-long:") else SHORT_SHA256,
            "conversion_sha256": plan["conversion_sha256"]}
    budgeted = campaign.item_provider(meta, root, max_calls=6, max_output=OUTPUT_BUDGET, context=CONTEXT_LIMIT)
    engine, memory, conversion = None, None, None
    try:
        tools = []
        if arm == "vestige":
            engine = HookedRealEngine(Path(__file__).resolve().parents[2], root / "native",
                         manifest_path=campaign.runs / "build-manifest.json",
                         hook_model_callback=budgeted.hook_callback(lambda: engine))
            await asyncio.to_thread(engine.__enter__)
            conversion = await asyncio.to_thread(admit_graph, engine, item.history)
            atomic_write(root / "conversion.json", canonical(conversion) + "\n")
            tools = [{"type": "function", "function": {"name": t["name"],
                      "description": t.get("description", ""), "parameters": t["inputSchema"]}}
                     for t in engine.tools]
        elif arm in {"bm25", "embedding"}:
            klass = BM25Memory if arm == "bm25" else BGEMemory
            memory = await asyncio.to_thread(klass, item.history)
            tools = [SEARCH_TOOL]
        question = item.full_prompt if arm == "bare" else item.question
        messages = [{"role": "system", "content": system_prompt(arm)}, {"role": "user", "content": question}]
        initial = list(messages)
        if engine:
            contexts = await asyncio.to_thread(engine.begin_turn, question)
            engine.hooks.pending_context.clear()
            messages.extend({"role": "user", "content": text} for text in contexts)
        response = None
        for iteration in range(6):
            if budgeted.remaining_calls <= 0 or budgeted.remaining_output <= 0:
                raise RuntimeError("aggregate solver-plus-hook budget exhausted")
            result = await budgeted.chat(messages, tools=tools,
                         max_tokens=min(512, budgeted.remaining_output), log=root / "api.jsonl",
                         identity={**meta, "iteration": iteration})
            message = result["message"]
            messages.append({k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}})
            calls = message.get("tool_calls") or []
            if not calls:
                if result["finish_reason"] == "length":
                    raise RuntimeError("GraphWalks response truncated at equal aggregate budget")
                response = message.get("content") or ""
                if engine:
                    stop = await asyncio.to_thread(engine.finish_turn, messages)
                    if stop["blocked"]:
                        messages.append({"role": "user", "content": "\n".join(stop["reasons"])})
                        response = None
                        continue
                break
            for call in calls:
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])
                if engine:
                    output = await asyncio.to_thread(engine.call, name, arguments)
                elif memory and name == "memory_search":
                    output = await asyncio.to_thread(memory.search, arguments["query"],
                                                    min(10, max(1, arguments.get("k", 5))))
                else:
                    raise RuntimeError("model requested an unavailable graph-memory tool")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": canonical(output)})
                if engine and engine.hooks.pending_context:
                    messages.extend({"role": "user", "content": text} for text in engine.hooks.pending_context)
                    engine.hooks.pending_context.clear()
        if response is None:
            raise RuntimeError("GraphWalks has no final model answer within equal aggregate budget")
        graph_hash = hashlib.sha256(canonical(item.history).encode()).hexdigest()
        return {**meta, "status": "complete", "response": response, "prompt": initial,
                "transcript": messages, "score": None, "model": MODEL,
                "usage": budgeted.usage, "cost": budgeted.cost, "finished_at": utc_now(),
                "engine": engine.identity if engine else None, "conversion": conversion,
                "group_id": graph_hash, "category": item.category,
                "raw_log": str(root / "api.jsonl"),
                "graph_placement": "official full context" if arm == "bare" else "source graph through memory tools"}
    finally:
        if engine:
            await asyncio.to_thread(engine.__exit__, None, None, None)


async def run_graphwalks(campaign):
    from .provider import MODEL
    from .prompts import audit_prompts
    from .runtime import (BudgetExceeded, Checkpoint, append_jsonl, atomic_write, canonical,
                          record_key, shard_id, strict_merge, utc_now, write_shard_map)
    from .run import ARMS, ci

    qualification = campaign.runs / "QUALIFICATION.json"
    if not qualification.exists():
        raise RuntimeError("HOOK_AUDIT_REQUIRED: GraphWalks cannot run before hook qualification")
    qualified = json.loads(qualification.read_text())
    if qualified.get("status") != "qualified" or qualified.get("vestige_arm") != "all_applicable_hooks":
        raise RuntimeError("HOOK_AUDIT_REQUIRED: GraphWalks sole Vestige arm needs all applicable hooks")
    if not hasattr(campaign, "item_provider"):
        raise RuntimeError("aggregate shared solver-plus-hook provider is required")
    items = await asyncio.to_thread(load_graphwalks, campaign.runs / "upstream")
    root = campaign.runs / "graphwalks"
    root.mkdir(parents=True, exist_ok=True)
    prompts = audit_prompts()
    identity = {"benchmark": "graphwalks", **verify_sources(campaign.runs / "upstream"),
                "binary_sha256": campaign.manifest["binary_sha256"], "model": MODEL,
                "hooks_manifest_sha256": qualified["hooks_manifest_sha256"], "prompts": prompts,
                "limits": [6, OUTPUT_BUDGET, CONTEXT_LIMIT], "trials": 1, "scope": "full1150 items from both official partitions",
                "conversion": "raw-edge-incidence-evidence-of-v1",
                "placement_exception": "Full official rawgraph in bare userprompt; otherarms receive identical official instruction/examples/operation with rawsourcegraph in memorytools."}
    atomic_write(root / "identity.json", canonical(identity) + "\n")
    write_shard_map(root / "shards.json", [i.id for i in items], workers=10, seed=20261101)
    checkpoints = {arm: [Checkpoint(root / arm / f"worker-{k}.jsonl", {**identity, "arm": arm})
                         for k in range(10)] for arm in ARMS}
    for arm in ARMS:
        campaign.progress["graphwalks", arm] = {"benchmark": "graphwalks", "arm": arm,
                   "done": sum(len(c.done) for c in checkpoints[arm]), "total": len(items), "errors": 0}

    async def one(item, arm, attempt=0):
        worker = shard_id(item.id)
        cp = checkpoints[arm][worker]
        key = {"item_id": item.id, "trial": 0}
        if cp.contains(key):
            return cp.done[record_key(key)]
        try:
            row = await solve_graphwalks(campaign, item, arm, 0, worker)
            cp.append(row)
            campaign.progress["graphwalks", arm]["done"] += 1
            return row
        except BudgetExceeded:
            raise
        except Exception as exc:
            cp.append_failure({**key, "attempt": attempt, "error_type": type(exc).__name__, "error": str(exc)[:500]})
            campaign.progress["graphwalks", arm]["errors"] += 1
            raise

    preflight_count = getattr(campaign, "preflight_count", None)
    if preflight_count:
        selected = random.Random(20261101).sample(items, min(int(preflight_count), len(items)))
        atomic_write(root / "preflight-source-map.json", canonical({
            "mode": "integrity_raw_preflight_only", "seed": 20261101,
            "official_full_set": [item.id for item in items],
            "preflight_item_ids": [item.id for item in selected],
            "dataset_hashes": identity["dataset_hashes"], "official_denominator": len(items),
            "gold_resolver": "load_graphwalks(...).scorer_payload.answer_nodes",
        }) + "\n")
        async def preflight_worker(arm, worker_id):
            for item in selected:
                if shard_id(item.id) != worker_id:
                    continue
                try:
                    await one(item, arm)
                except BudgetExceeded:
                    return
                except Exception:
                    pass  # exact error retained; audit cannot count a success
        await asyncio.gather(*(preflight_worker(arm, k) for arm in ARMS for k in range(10)))
        return {"status": "raw_preflight_only", "official_scoring": False,
                "official_denominator": len(items), "selected_ids": [item.id for item in selected]}

    from .integrity import audit, require
    require(audit(Path(__file__).resolve().parents[2], campaign.runs, stage="score"), "graphwalks", stage="score")

    projection_path = root / "projection.json"
    if not projection_path.exists():
        # First fixed publisher row, before any outcome. No source graph is
        # admitted for the other749 items before this cost gate opens.
        calibration = [await one(items[0], arm) for arm in ARMS]
        observed = sum(row["cost"] for row in calibration)
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")
        full_tokens = await asyncio.to_thread(lambda: sum(len(encoding.encode(item.full_prompt)) for item in items))
        projected = observed * len(items)
        projection = {"first_item": items[0].id, "calibration": [{"arm": r["arm"], "cost": r["cost"]} for r in calibration],
                      "projected_total": projected, "remaining": campaign.ledger.snapshot()["remaining"],
                      "full_context_input_tokens": full_tokens,
                      "bare_full_context_uncached_input_cost_floor": full_tokens * 0.20 / 1e6,
                      "method": "first fixed official item/arm actual solver+hook usage; naive extrapolation; graph sizes vary substantially"}
        projection["projected_total"] = max(projected, projection["bare_full_context_uncached_input_cost_floor"])
        atomic_write(projection_path, canonical(projection) + "\n")
        print("graphwalks PROJECTED_COST", canonical(projection), flush=True)
    projection = json.loads(projection_path.read_text())
    # Owner correction mem-00000000000b0be3: projection is a forecast only.
    # Keep the full frozen denominator and launch bounded work until the real
    # shared pre-call reservation cap stops new requests.

    async def worker(arm, worker_id):
        failed = []
        for item in items:
            if shard_id(item.id) != worker_id:
                continue
            try:
                await one(item, arm)
            except BudgetExceeded:
                return
            except Exception:
                failed.append(item)
        for item in failed:
            try:
                await one(item, arm, 1)
            except BudgetExceeded:
                return
            except Exception:
                pass  # errors persist and the strict merge refuses any gap

    await asyncio.gather(*(worker(arm, k) for arm in ARMS for k in range(10)))
    merged_by_arm = {}
    for arm in ARMS:
        expected = [record_key({"item_id": item.id, "trial": 0}) for item in items]
        merged_by_arm[arm] = strict_merge([cp.path for cp in checkpoints[arm]], expected, root / arm / "merged.jsonl")
    # Independent owner-required20-item/arm integrity audit precedes every
    # official scoring pass and RESULTS write. Missing audit stays pending.
    try:
        from .integrity import audit, require
        integrity = audit(Path(__file__).resolve().parents[2], campaign.runs, stage="scorer")
    except ImportError:
        atomic_write(root / "PENDING.json", canonical({"reason": "integrity_audit_not_available"}) + "\n")
        return
    require(integrity, "graphwalks", stage="scorer")
    summaries = []
    for arm in ARMS:
        merged = merged_by_arm[arm]
        approved = set(integrity.get("approved_run_identities", {}).get("graphwalks", {}).get(arm, []))
        if not approved or any(row["run_identity"] not in approved for row in merged):
            raise RuntimeError("GraphWalks scoring includes identities outside independent audit approval")
        cache = root / arm / "official-scores.json"
        digest = hashlib.sha256(canonical({"identity": identity, "arm": arm,
                    "predictions": {r["item_id"]: r["response"] for r in merged}}).encode()).hexdigest()
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved["identity"] != digest:
                raise RuntimeError("GraphWalks cached official scorer identity mismatch")
            scores = saved["scores"]
        else:
            scores = score_graphwalks(campaign.runs / "upstream", items, {r["item_id"]: r["response"] for r in merged})
            atomic_write(cache, canonical({"identity": digest, "scores": scores}) + "\n")
        values = [s["f1"] for s in scores]
        import sqlite3
        with sqlite3.connect(campaign.runs / "budget.db") as db:
            actual, reserved = db.execute(
                "SELECT COALESCE(SUM(actual),0), COALESCE(SUM(CASE WHEN state='reserved' THEN reserved ELSE 0 END),0) "
                "FROM calls WHERE id LIKE ?", (f"graphwalks/{arm}/%",)
            ).fetchone()
        summaries.append({"arm": arm, "mean": sum(values) / len(values),
            "ci95": ci(values, [r["group_id"] for r in merged]), "n": len(values), "trials": 1,
            "model": MODEL, "cost": sum(r["cost"] for r in merged),
            "cost_all_metered_calls": actual / 1e9, "unreconciled_reservations": reserved / 1e9,
            "binary_sha256": campaign.manifest["binary_sha256"], "dataset_sha256": SHORT_SHA256,
            "dataset_hashes": {SHORT_FILE: SHORT_SHA256, LONG_FILE: LONG_SHA256},
            "failed_to_parse": sum(s["failed_to_parse"] for s in scores)})
    atomic_write(root / "results.json", canonical(summaries) + "\n")
    write_graphwalks_results(root, summaries)
    return summaries


def write_graphwalks_results(root: Path, summaries):
    from .runtime import atomic_write
    from .integrity import audit, require
    require(audit(Path(__file__).resolve().parents[2], root.parent, stage="report"), "graphwalks", stage="report")
    lines = ["# GraphWalks real-engine results", "",
             "Official set F1, executed from the pinned publisher README code. "
             "95% bootstrap confidence intervals cluster items sharing an identical raw source graph. "
             "All1150 items in both official partitions use one trial and a common131072-token context limit, "
             "2048 aggregate output tokens and6 aggregate solver-plus-hook calls.", "",
             "| Arm | Mean F1 | 95% CI | n | Trials | Model | All settled API cost | Binary SHA-256 | Dataset SHA-256 |",
             "|---|---:|---|---:|---:|---|---:|---|---|"]
    for row in summaries:
        lines.append(f"| {row['arm']} | {row['mean']:.4f} | {row['ci95']} | {row['n']} | {row['trials']} "
                     f"| {row['model']} | ${row['cost_all_metered_calls']:.6f} | `{row['binary_sha256']}` | `{row['dataset_sha256']}` |")
    lines += ["", "The bare arm receives the complete original official graph prompt. "
              "The other arms receive the same official worked examples, operation and answer-format wording, "
              "with the actual raw graph supplied through their memory tools. "
              "This graph-placement difference is the explicit full-context-versus-memory comparison. "
              "System prompts differ only by one neutral tools line.", "",
              "Vestige stores raw endpoint identifiers and raw directed-edge records through normal admitted MCP writes. "
              "Each edge record has receipted evidence-of links to its two endpoints. "
              "This is a source-fact incidence representation: the model reads raw directions and computes the requested operation. "
              "The host computes no reachability, supplies no answer-derived links, and never injects target answer nodes. "
              "The native GhostLink map interface is limited to depth3 and200nodes per call. "
              "Conversion hashes and native admission receipts are retained per item.", "",
              "All applicable source-wired Vestige hooks run in the sole Vestige arm. Their model calls use "
              "the same nano model, shared global ledger and the same aggregate item budget as solver calls. "
              "Failed attempts and outstanding reservations remain visible in raw logs and the global ledger.", "",
              "The full official denominator includes750 short-partition and400 long-partition rows. "
              "Rows exceeding the common context cap remain explicit unfinished errors; no graph is truncated "
              "or removed from the official denominator. Long-partition SHA-256: `" + LONG_SHA256 + "`. "
              "Published, different model or judge: no Mem0/Zep/Letta GraphWalks number is inserted without a verified matching source.", ""]
    atomic_write(root / "RESULTS.md", "\n".join(lines))

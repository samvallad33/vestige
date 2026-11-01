"""Live seven-day PMBench through its pinned official simulator.

Only the model transport and a single neutral tool-description line are
adapted. State visibility, observation generation, action handling, updates,
and score_log remain the official implementation. No scenario object enters
the model bridge: it receives only the simulator's visible chat messages.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
from pathlib import Path
import uuid

from .extra_datasets import (
    PMBENCH_DATA_SHA256, PMBENCH_REVISION, _import_file, _pmbench_repo,
    load_pmbench_scenario, score_pmbench,
)
from .prompts import TOOL_LINES
from .provider import MODEL
from .runtime import (
    BudgetExceeded, Checkpoint, append_jsonl, atomic_write, canonical,
    record_key, shard_id, strict_merge, utc_now, write_shard_map,
)

ARMS = ("bare", "bm25", "embedding", "vestige")
TRIALS = 3
WORKERS = 10
MAX_ACTION_REQUESTS = 640
MAX_TOOL_ITERS = 6
ACTION_OUTPUT_BUDGET = 2048
CONTEXT_LIMIT = 32768


class PMActionError(RuntimeError):
    """An explicit incomplete trial; never triggers upstream oracle fallback."""


class PreflightComplete(RuntimeError):
    """Stop raw collection after its declared count, without official scoring."""


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _token_count(value):
    import tiktoken
    # Official openai/tiktoken main model.py maps the gpt-5 prefix to
    # o200k_base (checked 2026-11-01). Version0.12.0 predates that prefix.
    return len(tiktoken.get_encoding("o200k_base").encode(canonical(value)))


def simulator_module(root):
    """A private module instance per trial avoids cross-worker monkeypatching."""
    load_pmbench_scenario(root)  # source + dataset hashes checked before import
    return _import_file(_pmbench_repo(root) / "sim/pm_bench.py", "_pmbench_trial_" + uuid.uuid4().hex)


def action_prompt(simulator, arm, **kwargs):
    return simulator.build_llm_system_prompt(**kwargs) + "\n" + TOOL_LINES[arm]


def audit_pmbench_prompts(upstream):
    sim = simulator_module(upstream)
    scenario = load_pmbench_scenario(upstream)
    kwargs = {"show_task_legend": False, "state_channels": sim.list_state_channels(scenario),
              "allow_heartbeat": False, "auto_heartbeat_minutes": None,
              "heartbeat_message_mode": "channel_query"}
    original = sim.build_llm_system_prompt(**kwargs)
    prompts = {arm: action_prompt(sim, arm, **kwargs) for arm in ARMS}
    for arm, prompt in prompts.items():
        if prompt.rsplit("\n", 1) != [original, TOOL_LINES[arm]]:
            raise PMActionError("PMBench prompt differs beyond its neutral tools line")
        for forbidden in ("session-window", "app/events.py", "EventSource", "idle_timeout", "Specific Guidance"):
            if forbidden.lower() in prompt.lower():
                raise PMActionError("task-specific guidance in PMBench system prompt")
    return {arm: {"text": prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            for arm, prompt in prompts.items()}


class PMActionBridge:
    def __init__(self, campaign, simulator, arm, trial, worker, root, engine=None,
                 initial_budget=None, hook_callback=None):
        self.campaign, self.simulator, self.arm = campaign, simulator, arm
        self.trial, self.worker, self.root, self.engine = trial, worker, Path(root), engine
        self.message_cursor = 0
        self.observed = []
        self.requests = 0
        self.current_step = None
        self.last_record = None
        self.initial_budget = initial_budget
        self.hook_callback = hook_callback
        self.budgets = [initial_budget] if initial_budget is not None else []

    @property
    def cost(self):
        return sum(budget.cost for budget in self.budgets)

    @property
    def usage(self):
        return [usage for budget in self.budgets for usage in budget.usage]

    async def action(self, messages):
        """Return a model-selected action from visible events only."""
        from .retrieval import BM25Memory, BGEMemory
        from .run import SEARCH_TOOL

        if self.requests >= MAX_ACTION_REQUESTS:
            raise PMActionError("equal per-trial simulator action-request budget exhausted")
        self.requests += 1
        meta = {"benchmark": "pmbench", "arm": self.arm, "trial": self.trial,
                "item_id": f"pmbench:official-week:trial-{self.trial}",
                "seed": self.trial, "shard": self.worker, "worker_id": self.worker,
                "binary_sha256": self.campaign.manifest["binary_sha256"],
                "action_request": self.requests}
        if self.current_step:
            meta.update(day=self.current_step[0], step_id=self.current_step[1])
            meta["group_id"] = meta["item_id"]
            meta["item_id"] = pmbench_step_id(self.trial, *self.current_step)
        if self.requests == 1 and self.initial_budget is not None:
            budgeted = self.initial_budget
            budgeted.meta.update(meta)
        else:
            budgeted = self.campaign.item_provider(meta, self.root,
                       max_calls=MAX_TOOL_ITERS, max_output=ACTION_OUTPUT_BUDGET, context=CONTEXT_LIMIT)
            self.budgets.append(budgeted)
        if self.hook_callback is not None:
            # One engine persists throughout the week. The source hook bridge
            # uses the same new per-action aggregate budget as the solver.
            self.hook_callback.bound = budgeted
        # The official simulator appends to one dialogue throughout the week.
        # Read only newly observed user/assistant messages, never hidden state.
        fresh = messages[self.message_cursor:]
        self.message_cursor = len(messages)
        new_records = [f"{m['role']}: {m.get('content') or ''}" for m in fresh if m['role'] != 'system']
        self.observed.extend(new_records)
        tools, memory = [], None
        if self.engine:
            await asyncio.to_thread(self.engine.ingest_history, new_records)
            allowed = {"recall", "memory", "session_start", "smart_ingest", "causal_walk"}
            tools = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t["inputSchema"],
            }} for t in self.engine.tools if t["name"] in allowed]
        elif self.arm in {"bm25", "embedding"}:
            klass = BM25Memory if self.arm == "bm25" else BGEMemory
            memory = await asyncio.to_thread(klass, self.observed)
            tools = [SEARCH_TOOL]
        transcript = copy.deepcopy(messages)
        if self.engine:
            contexts = await asyncio.to_thread(self.engine.begin_turn, messages[-1].get("content") or "")
            self.engine.hooks.pending_context.clear()
            transcript.extend({"role": "user", "content": text} for text in contexts)
        for iteration in range(MAX_TOOL_ITERS):
            if _token_count({"messages": transcript, "tools": tools}) > CONTEXT_LIMIT:
                raise PMActionError("equal context budget exhausted; no hidden truncation")
            if budgeted.remaining_calls <= 0 or budgeted.remaining_output <= 0:
                raise PMActionError("equal aggregate action solver-plus-hook budget exhausted")
            result = await budgeted.chat(
                transcript, tools=tools, max_tokens=min(512, budgeted.remaining_output),
                log=self.root / "api.jsonl", identity={
                    "benchmark": "pmbench", "arm": self.arm, "trial": self.trial,
                    "seed": self.trial, "shard": self.worker, "worker_id": self.worker,
                    "binary_sha256": self.campaign.manifest["binary_sha256"],
                    "action_request": self.requests, "iteration": iteration,
                },
            )
            message = result["message"]
            transcript.append({k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}})
            calls = message.get("tool_calls") or []
            if not calls:
                raw = message.get("content") or ""
                if result["finish_reason"] == "length":
                    raise PMActionError("PMBench action truncated at equal output budget")
                try:
                    action = self.simulator.parse_json_payload(raw)
                except Exception as exc:
                    # Upstream InvalidModelResponseError would invoke an oracle
                    # fallback using due_now. This distinct error propagates out
                    # of official run_llm and leaves the trial unfinished.
                    raise PMActionError("invalid model action; oracle fallback prohibited") from exc
                if self.engine:
                    stop = await asyncio.to_thread(self.engine.finish_turn, transcript)
                    if stop["blocked"]:
                        transcript.append({"role": "user", "content": "\n".join(stop["reasons"])})
                        continue
                self.last_record = {
                    "timestamp": utc_now(), "action_request": self.requests,
                    "arm": self.arm, "trial": self.trial, "seed": self.trial,
                    "shard": self.worker, "worker_id": self.worker,
                    "binary_sha256": self.campaign.manifest["binary_sha256"],
                    "visible_messages": messages, "model_response": raw,
                    "action": action, "cost_so_far": self.cost,
                    "transcript": transcript,
                }
                append_jsonl(self.root / "actions.jsonl", self.last_record)
                return action
            for call in calls:
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])
                if self.engine:
                    output = await asyncio.to_thread(self.engine.call, name, arguments)
                elif memory and name == "memory_search":
                    output = await asyncio.to_thread(memory.search, arguments["query"],
                                                    min(10, max(1, arguments.get("k", 5))))
                else:
                    raise PMActionError("model requested unavailable memory tool")
                transcript.append({"role": "tool", "tool_call_id": call["id"], "content": canonical(output)})
                if self.engine and self.engine.hooks.pending_context:
                    transcript.extend({"role": "user", "content": text} for text in self.engine.hooks.pending_context)
                    self.engine.hooks.pending_context.clear()
            if budgeted.remaining_output <= 0:
                raise PMActionError("equal aggregate action output budget exhausted")
        raise PMActionError("equal memory-tool iteration budget exhausted")


async def run_pmbench_trial(campaign, arm, trial, worker, *, preflight_steps=None, accepted_step_callback=None):
    """Run one independent full week. Parent owns permission and shared budget."""
    from .engine import HookedRealEngine

    upstream = campaign.runs / "upstream"
    scenario = await asyncio.to_thread(load_pmbench_scenario, upstream)
    sim = await asyncio.to_thread(simulator_module, upstream)
    item_id = f"pmbench:official-week:trial-{trial}"
    root = campaign.runs / "pmbench" / arm / f"worker-{worker}" / f"trial-{trial}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    engine = None
    loop = asyncio.get_running_loop()
    meta = {"benchmark": "pmbench", "arm": arm, "trial": trial, "seed": trial,
            "worker_id": worker, "shard": worker, "item_id": item_id,
            "binary_sha256": campaign.manifest["binary_sha256"], "action_request": 1}
    initial_budget = campaign.item_provider(meta, root, max_calls=MAX_TOOL_ITERS,
                                           max_output=ACTION_OUTPUT_BUDGET, context=CONTEXT_LIMIT)
    hook_callback = initial_budget.hook_callback(lambda: engine) if arm == "vestige" else None
    try:
        if arm == "vestige":
            engine = await asyncio.to_thread(HookedRealEngine, Path(__file__).resolve().parents[2], root / "native",
                                            manifest_path=campaign.runs / "build-manifest.json",
                                            hook_model_callback=hook_callback)
            await asyncio.to_thread(engine.__enter__)
        bridge = PMActionBridge(campaign, sim, arm, trial, worker, root, engine,
                                initial_budget=initial_budget, hook_callback=hook_callback)
        step_keys = [(day["name"], step["id"]) for day in scenario["days"] for step in day["steps"]]
        accepted = []
        cost_cursor, usage_cursor = 0.0, 0
        original_complete = sim.apply_runtime_completions

        def complete(*args, **kwargs):
            nonlocal cost_cursor, usage_cursor
            result = original_complete(*args, **kwargs)
            day, step_id = step_keys[len(accepted)]
            raw = bridge.last_record
            if raw is None:
                raise PMActionError("official runtime completed a step without a model-selected action")
            row = {"benchmark": "pmbench", "item_id": pmbench_step_id(trial, day, step_id),
                   "group_id": item_id, "trial": trial, "seed": trial, "day": day, "step_id": step_id,
                   "arm": arm, "worker_id": worker, "shard": worker, "status": "complete",
                   "phase": "integrity_preflight" if preflight_steps else "full_week_step",
                   "binary_sha256": campaign.manifest["binary_sha256"], "model": MODEL,
                   "dataset_sha256": PMBENCH_DATA_SHA256, "score": None,
                   "prompt": raw["visible_messages"], "transcript": raw["transcript"],
                   "response": raw["model_response"], "cost": bridge.cost - cost_cursor,
                   "usage": bridge.usage[usage_cursor:], "action_request": bridge.requests,
                   "raw_log": str(root / "api.jsonl"), "timestamp": utc_now(),
                   "engine": engine.identity if engine else None}
            cost_cursor, usage_cursor = bridge.cost, len(bridge.usage)
            accepted.append(row)
            append_jsonl(root / "accepted-steps.jsonl", row)
            if accepted_step_callback:
                accepted_step_callback(row)
            if preflight_steps and len(accepted) >= preflight_steps:
                raise PreflightComplete("declared distinct-step raw preflight collection complete")
            return result

        sim.apply_runtime_completions = complete
        original_prompt = sim.build_llm_system_prompt
        sim.build_llm_system_prompt = lambda *a, **kw: original_prompt(*a, **kw) + "\n" + TOOL_LINES[arm]
        # Transport-only adapters; the original provider helper is never called.
        sim.build_llm_client = lambda *a, **kw: None

        def request(client, model, messages, *args, **kwargs):
            bridge.current_step = step_keys[len(accepted)]
            return asyncio.run_coroutine_threadsafe(bridge.action(copy.deepcopy(messages)), loop).result()

        sim.request_model_action = request
        sim.print = lambda *a, **kw: None  # per-module console output only
        try:
            entries = await asyncio.to_thread(
                sim.run_llm, scenario=scenario, log_path=str(root / "trajectory.jsonl"), model=MODEL,
                env_path=None, max_time_requests=5, backend="openai", base_url=None, api_key=None,
                prompt_log_path=None, show_task_legend=False, out_dir=None,
                enable_heartbeat=False, auto_heartbeat_minutes=None, max_tokens=512,
            )
        except PreflightComplete:
            return {"status": "raw_preflight_complete", "benchmark": "pmbench", "arm": arm,
                    "trial": trial, "accepted_steps": len(accepted), "item_ids": [r["item_id"] for r in accepted],
                    "cost": bridge.cost, "usage": bridge.usage, "score": None,
                    "raw_log": str(root / "api.jsonl"), "official_denominator": "3 whole80-step weeks/arm unchanged"}
        # Validate coverage now; official scoring happens after shard merge.
        expected = {(d["name"], s["id"]) for d in scenario["days"] for s in d["steps"]}
        actual = [(e["day"], e["step_id"]) for e in entries]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise PMActionError("official simulator did not complete all 80 unique steps")
        return {
            "benchmark": "pmbench", "item_id": item_id, "trial": trial, "seed": trial,
            "arm": arm, "shard": worker, "worker_id": worker, "status": "complete",
            "binary_sha256": campaign.manifest["binary_sha256"], "model": MODEL,
            "dataset_sha256": PMBENCH_DATA_SHA256, "dataset_revision": PMBENCH_REVISION,
            "started_at": started, "finished_at": utc_now(), "entries": entries,
            "cost": bridge.cost, "usage": bridge.usage, "action_requests": bridge.requests,
            "engine": engine.identity if engine else None, "raw_log": str(root / "api.jsonl"),
            "score": None,
        }
    finally:
        if engine:
            await asyncio.to_thread(engine.__exit__, None, None, None)


async def run_pmbench(campaign):
    """Three whole-week trials per arm, checkpointed and globally metered."""
    name, root = "pmbench", campaign.runs / "pmbench"
    root.mkdir(parents=True, exist_ok=True)
    qualification = campaign.runs / "QUALIFICATION.json"
    if not qualification.exists():
        raise RuntimeError("HOOK_AUDIT_REQUIRED: PMBench cannot run before full hook qualification")
    qualified = json.loads(qualification.read_text())
    if qualified.get("status") != "qualified" or qualified.get("vestige_arm") != "all_applicable_hooks":
        raise RuntimeError("HOOK_AUDIT_REQUIRED: PMBench sole Vestige arm needs all applicable hooks")
    if not hasattr(campaign, "item_provider"):
        raise RuntimeError("PMBench requires aggregate shared solver-plus-hook provider")
    await asyncio.to_thread(load_pmbench_scenario, campaign.runs / "upstream")
    identities = [f"pmbench:official-week:trial-{t}" for t in range(TRIALS)]
    write_shard_map(root / "shards.json", identities, workers=WORKERS, seed=20261101)
    identity = {"benchmark": name, "dataset_sha256": PMBENCH_DATA_SHA256,
                "binary_sha256": campaign.manifest["binary_sha256"], "model": MODEL,
                "hooks_manifest_sha256": qualified["hooks_manifest_sha256"],
                "trials": TRIALS, "tools_lines": TOOL_LINES, "official_scenario_steps": 80,
                "limits": [MAX_ACTION_REQUESTS, MAX_TOOL_ITERS, ACTION_OUTPUT_BUDGET, CONTEXT_LIMIT],
                "unit": "full seven-day trajectory; days and steps are not independent shards"}
    prompts = audit_pmbench_prompts(campaign.runs / "upstream")
    identity["prompt_hashes"] = {arm: data["sha256"] for arm, data in prompts.items()}
    atomic_write(root / "prompts.json", canonical(prompts) + "\n")
    checkpoints = {arm: [Checkpoint(root / arm / f"worker-{k}.jsonl", identity) for k in range(WORKERS)]
                   for arm in ARMS}
    for arm in ARMS:
        campaign.progress[name, arm] = {"benchmark": name, "arm": arm,
                                        "done": sum(len(c.done) for c in checkpoints[arm]),
                                        "total": TRIALS, "errors": 0}
    atomic_write(root / "identity.json", canonical(identity) + "\n")
    preflight_count = getattr(campaign, "preflight_count", None)
    if preflight_count:
        count = min(int(preflight_count), 80)
        scenario = load_pmbench_scenario(campaign.runs / "upstream")
        keys = [(day["name"], step["id"]) for day in scenario["days"] for step in day["steps"]]
        source_map = {"mode": "integrity_raw_preflight_only", "dataset_sha256": PMBENCH_DATA_SHA256,
                      "official_full_set": [pmbench_step_id(t, day, step) for t in range(TRIALS) for day, step in keys],
                      "preflight_item_ids": [pmbench_step_id(0, day, step) for day, step in keys[:count]],
                      "full_trials": TRIALS, "full_steps_per_trial": 80, "gold_resolver": "resolve_pmbench_label"}
        source_map["audit_sample_seed"] = 20261101
        source_map["audit_sample_ids"] = random.Random(20261101).sample(source_map["preflight_item_ids"], count)
        source_map["sampling_note"] = "Fixed-seed audit order over the completed chronological prefix; no simulator state is skipped and no full-run denominator is reduced."
        atomic_write(root / "preflight-source-map.json", canonical(source_map) + "\n")
        async def preflight_arm(arm):
            worker = shard_id(identities[0], workers=WORKERS)
            cp = Checkpoint(root / arm / "preflight" / f"worker-{worker}.jsonl", {**identity, "arm": arm,
                            "phase": "integrity_preflight", "count": count})
            if len(cp.done) >= count:
                return {"arm": arm, "status": "raw_preflight_complete", "accepted_steps": len(cp.done)}
            # A partial week is not resumed out of chronology. Preserve old
            # observations and use a fresh attempt checkpoint for an exact run.
            if cp.done:
                cp = Checkpoint(root / arm / "preflight" / uuid.uuid4().hex[:8] / f"worker-{worker}.jsonl",
                                {**identity, "arm": arm, "phase": "integrity_preflight", "count": count})
            campaign.progress[name, arm].update(benchmark="pmbench (raw official step/trial rows)",
                                               total=80 * TRIALS, done=len(cp.done))
            def checkpoint(row):
                cp.append(row)
                campaign.progress[name, arm]["done"] = len(cp.done)
            try:
                return await run_pmbench_trial(campaign, arm, 0, worker,
                                preflight_steps=count, accepted_step_callback=checkpoint)
            except Exception as exc:
                append_jsonl(root / "preflight-errors.jsonl", {"arm": arm, "error_type": type(exc).__name__,
                             "error": str(exc)[:500], "timestamp": utc_now()})
                return {"arm": arm, "status": "error", "error": str(exc)[:500]}
        outcomes = await asyncio.gather(*(preflight_arm(arm) for arm in ARMS))
        atomic_write(root / "preflight-summary.json", canonical(outcomes) + "\n")
        return {"status": "raw_preflight_only", "outcomes": outcomes, "official_scoring": False}

    from .integrity import audit, require
    require(audit(Path(__file__).resolve().parents[2], campaign.runs, stage="score"), "pmbench", stage="score")

    async def one(arm, trial, attempt=0):
        worker = shard_id(identities[trial], workers=WORKERS)
        cp = checkpoints[arm][worker]
        key = {"item_id": identities[trial], "trial": trial}
        if cp.contains(key):
            return cp.done[record_key(key)]
        try:
            row = await run_pmbench_trial(campaign, arm, trial, worker)
            cp.append(row)
            campaign.progress[name, arm]["done"] += 1
            return row
        except BudgetExceeded:
            raise
        except Exception as exc:
            cp.append_failure({**key, "attempt": attempt, "error": str(exc)[:500], "error_type": type(exc).__name__})
            campaign.progress[name, arm]["errors"] += 1
            raise

    projection_path = root / "projection.json"
    if not projection_path.exists():
        # A PMBench item IS a whole official week, so the pilot never uses a
        # shortened day/sample to estimate a different task.
        calibration = [await one(arm, 0) for arm in ARMS]
        projected = sum(row["cost"] for row in calibration) * TRIALS
        projection = {"unit": "one whole seven-day week per arm", "trials": TRIALS,
                      "calibration": [{"arm": row["arm"], "cost": row["cost"]} for row in calibration],
                      "projected_total": projected, "remaining": campaign.ledger.snapshot()["remaining"]}
        atomic_write(projection_path, canonical(projection) + "\n")
        print("pmbench PROJECTED_COST", canonical(projection), flush=True)
    projection = json.loads(projection_path.read_text())
    # Projection is a forecast, not permission to shrink or skip the official
    # full set (owner correction mem-00000000000b0be3). Actual shared bounded
    # reservations stop new calls at the cap and preserve every unfinished id.

    async def worker(arm, worker_id):
        failures = []
        for trial in range(TRIALS):
            if shard_id(identities[trial], workers=WORKERS) != worker_id:
                continue
            try:
                await one(arm, trial)
            except BudgetExceeded:
                return
            except Exception:
                failures.append(trial)
        for trial in failures:
            try:
                await one(arm, trial, attempt=1)
            except BudgetExceeded:
                return
            except Exception:
                pass  # retained failure logs + merge rejection, never scored

    await asyncio.gather(*(worker(arm, k) for arm in ARMS for k in range(WORKERS)))
    merged_by_arm = {}
    for arm in ARMS:
        expected = [record_key({"item_id": identities[t], "trial": t}) for t in range(TRIALS)]
        merged_by_arm[arm] = strict_merge([cp.path for cp in checkpoints[arm]], expected, root / arm / "merged.jsonl")
    try:
        from .integrity import audit, require
        integrity = audit(Path(__file__).resolve().parents[2], campaign.runs, stage="scorer")
    except ImportError:
        atomic_write(root / "PENDING.json", canonical({"reason": "integrity_audit_not_available"}) + "\n")
        return
    require(integrity, "pmbench", stage="scorer")
    for arm in ARMS:
        merged = merged_by_arm[arm]
        approved = set(integrity.get("approved_run_identities", {}).get("pmbench", {}).get(arm, []))
        if not approved or any(row["run_identity"] not in approved for row in merged):
            raise PMActionError("PMBench scoring includes identities outside independent audit approval")
        scored = []
        for row in merged:
            cache = root / arm / f"official-score-trial-{row['trial']}.json"
            artifact_identity = _hash({"identity": identity, "trajectory": row["entries"]})
            if cache.exists():
                saved = json.loads(cache.read_text())
                if saved["artifact_identity"] != artifact_identity:
                    raise PMActionError("cached official PMBench score identity mismatch")
                result = saved["result"]
            else:
                result = score_pmbench(campaign.runs / "upstream", row["entries"])
                atomic_write(cache, canonical({"artifact_identity": artifact_identity, "result": result}) + "\n")
            scored.append({**row, "official_score": result})
        atomic_write(root / arm / "official-scores.json", canonical(scored) + "\n")
    write_results(campaign)
    return {"status": "complete", "arms": list(ARMS), "trials": TRIALS, "steps_per_trial": 80}


def write_results(campaign):
    """Official hit-rate denominator; bootstrap whole-week independent trials."""
    import numpy as np
    from .integrity import audit, require
    require(audit(Path(__file__).resolve().parents[2], campaign.runs, stage="report"), "pmbench", stage="report")
    root = campaign.runs / "pmbench"
    rows = []
    for arm in ARMS:
        scored = json.loads((root / arm / "official-scores.json").read_text())
        rates = []
        for trial in scored:
            result = trial["official_score"]
            # Same denominator used by the official report: summed by_type total.
            total = sum(counts["total"] for day in result["per_day"].values()
                        for counts in day["by_type"].values())
            if not total:
                raise PMActionError("official PMBench hit-rate denominator is zero")
            rates.append(result["summary"]["hit"] / total)
        rng = np.random.default_rng(20261101)
        means = np.mean(rng.choice(rates, size=(10000, len(rates)), replace=True), axis=1)
        lower, upper = np.quantile(means, [0.025, 0.975])
        # Include settled charges from failed attempts too, not only successes.
        cost = 0.0
        for log in (root / arm).rglob("api.jsonl"):
            with log.open() as f:
                for line in f:
                    event = json.loads(line)
                    if event.get("event") == "response":
                        cost += event["cost"]
        rows.append({"arm": arm, "mean": float(np.mean(rates)), "ci": [float(lower), float(upper)],
                     "n": len(rates), "steps": len(rates) * 80, "trials": [x["trial"] for x in scored],
                     "cost": cost, "model": MODEL})
    lines = ["# PMBench real-engine results", "",
             "Metric: official on-time hit rate, using the official by-type task denominator. "
             "The 95% bootstrap interval resamples whole seven-day trials (10,000 draws; seed 20261101). "
             "With three trials, uncertainty estimates are limited; steps are correlated within a week.", "",
             "| Arm | Mean hit rate | 95% CI | n whole weeks | Steps | Trials | Model | API cost |",
             "|---|---:|---|---:|---:|---|---|---:|"]
    for row in rows:
        lines.append(f"| {row['arm']} | {row['mean']:.4f} | [{row['ci'][0]:.4f}, {row['ci'][1]:.4f}] "
                     f"| {row['n']} | {row['steps']} | {row['trials']} | {row['model']} | ${row['cost']:.6f} |")
    lines += ["", f"Dataset revision: `{PMBENCH_REVISION}`.",
              f"Dataset SHA-256: `{PMBENCH_DATA_SHA256}`.",
              f"Engine binary SHA-256: `{campaign.manifest['binary_sha256']}`.", "",
              "The original seven-day simulator controls visible observations, hidden channels, updates and actions. "
              "The model transport uses the shared campaign budget and concurrency controls. "
              "All arm system prompts differ only by one neutral tools line. "
              "A model-format failure stops the trial; the upstream due-now oracle fallback is never used.", "",
              "Ten deterministic worker buckets are allocated per arm. Only three whole-week trial items exist, "
              "so at most three buckets contain work; days and steps are not independently sharded. "
              "Each Vestige week owns a separate real stdio child and store. "
              "The official scorer runs once per complete merged week, after the arm's three-trial merge; "
              "its result is cached by trajectory identity for restart.", "",
              "Raw provider requests/responses and intermediate actions are retained per attempt; "
              "completed whole weeks checkpoint immediately. A failed partial week retains its logs and retries "
              "as a new independent attempt. Official raw metrics are in each arm's official-scores.json.", "",
              "Published, different model or judge: no vendor numbers are inserted without a verified "
              "PMBench source and matching metric. No semantic judge is needed by the official replay scorer.", ""]
    atomic_write(root / "RESULTS.md", "\n".join(lines))
    atomic_write(root / "summary.json", canonical(rows) + "\n")


def pmbench_step_id(trial, day, step_id):
    return f"pmbench:trial-{trial}:{day}:{step_id}"


def resolve_pmbench_label(root, item_id):
    """Audit-only official label resolver; no solver path calls this helper."""
    scenario = load_pmbench_scenario(root)
    sim = simulator_module(root)
    for trial in range(TRIALS):
        for day in scenario["days"]:
            for step in day["steps"]:
                if pmbench_step_id(trial, day["name"], step["id"]) == item_id:
                    groundtruth = sim.compute_scenario_groundtruth(scenario)
                    return {"dataset_sha256": PMBENCH_DATA_SHA256, "trial": trial,
                            "day": day["name"], "step_id": step["id"],
                            "official_groundtruth": groundtruth["days"][day["name"]]["step_groundtruth"][step["id"]]}
    raise KeyError("unknown official PMBench step/trial identity")


def load_pmbench_audit_items(root):
    """Independent audit inputs for all240 official step/trial identities.

    This helper is never called by a solver. Hidden channel state and task
    criteria are absent from permitted history. Dynamic menus/query replies
    should be verified with replay_pmbench_visible using persisted real actions.
    """
    scenario = load_pmbench_scenario(root)
    sim = simulator_module(root)
    truth = sim.compute_scenario_groundtruth(scenario)
    system = sim.build_llm_system_prompt(False, sim.list_state_channels(scenario),
                 allow_heartbeat=False, auto_heartbeat_minutes=None, heartbeat_message_mode="channel_query")
    output = []
    for trial in range(TRIALS):
        permitted = [system, "\n".join(sim.DAILY_TASK_HEADER_LINES), *TOOL_LINES.values()]
        for day in scenario["days"]:
            permitted.append("\n".join([f"=== {day['name']} ==="] + day.get("start_instructions", [])))
            for step in day["steps"]:
                visible = str(step["text"]) + "\n" + "\n".join(step["options"])
                permitted.append(visible)
                output.append({"id": pmbench_step_id(trial, day["name"], step["id"]),
                    "group_id": f"pmbench:official-week:trial-{trial}", "question": visible,
                    "history": list(permitted), "category": "official_temporal_step",
                    "scorer_payload": {"groundtruth": truth["days"][day["name"]]["step_groundtruth"][step["id"]]},
                    "audit_source": {"dataset_sha256": PMBENCH_DATA_SHA256, "trial": trial,
                                     "day": day["name"], "step_id": step["id"]}})
    if len(output) != 240:
        raise PMActionError("official PMBench audit denominator must remain240 step/trial identities")
    return output


def expand_pmbench_week_records(weeks):
    """Project completed official whole-week records into their actual80 steps.

    Requires the independently persisted accepted-step transcript for each step;
    never invents a response or reconstructs one from an official gold label.
    """
    output = []
    for week in weeks:
        path = Path(week["raw_log"]).parent / "accepted-steps.jsonl"
        raw_steps = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        actual = {row["item_id"]: row for row in raw_steps}
        if len(actual) != len(raw_steps) or len(actual) != 80 or len(week["entries"]) != 80:
            raise PMActionError("completed PMBench week lacks80 unique actual step transcripts")
        for entry in week["entries"]:
            identity = pmbench_step_id(week["trial"], entry["day"], entry["step_id"])
            if identity not in actual:
                raise PMActionError("official PMBench trajectory step has no source transcript")
            output.append({**actual[identity], "run_identity": week["run_identity"],
                           "official_runtime_entry": entry, "week_item_id": week["item_id"]})
    return output


def replay_pmbench_visible(root, actions_path, arm):
    """Audit-only no-inference replay of actual model-selected actions.

    The unchanged official simulator regenerates each visible observation and
    query response. It receives recorded REAL model actions, never truth labels.
    This is provenance verification, not another benchmark run or score.
    """
    import tempfile
    scenario = load_pmbench_scenario(root)
    sim = simulator_module(root)
    actions = [json.loads(line) for line in Path(actions_path).read_text().splitlines() if line.strip()]
    calls = []
    original_prompt = sim.build_llm_system_prompt
    sim.build_llm_system_prompt = lambda *a, **kw: original_prompt(*a, **kw) + "\n" + TOOL_LINES[arm]
    sim.build_llm_client = lambda *a, **kw: None
    sim.print = lambda *a, **kw: None

    def request(client, model, messages, *a, **kw):
        if len(calls) >= len(actions):
            raise PreflightComplete("persisted actual action sequence exhausted")
        action = actions[len(calls)]
        calls.append({"action_request": len(calls) + 1, "messages": copy.deepcopy(messages),
                      "recorded_visible_messages": action["visible_messages"]})
        return sim.parse_json_payload(action["model_response"])

    sim.request_model_action = request
    with tempfile.TemporaryDirectory() as directory:
        try:
            sim.run_llm(scenario, str(Path(directory) / "replay.jsonl"), MODEL, None, 5,
                        "openai", None, None, show_task_legend=False, enable_heartbeat=False,
                        auto_heartbeat_minutes=None, max_tokens=512)
        except PreflightComplete:
            pass
    for row in calls:
        if row["messages"] != row["recorded_visible_messages"]:
            raise PMActionError("persisted PMBench visible model input differs from official simulator replay")
    return {"classification": "no_inference_source_provenance_replay_not_score", "calls": calls,
            "verified_action_requests": len(calls), "dataset_sha256": PMBENCH_DATA_SHA256}

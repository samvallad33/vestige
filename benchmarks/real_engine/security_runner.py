"""Full official security sets; five workers per arm on the shared campaign.

Preflight20 identifies inputs for integrity inspection, never a scored subset.
Every expected official id remains in the full manifest and incomplete ledger.
No RESULTS.md is written here. Final scoring is held until the campaign's final
integrity callback returns PASS and pins this exact runtime/adapters/merge.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import random
import uuid

from .provider import MODEL
from .runtime import (BudgetExceeded, Checkpoint, atomic_write, canonical, read_jsonl,
                      record_key, shard_id, strict_merge, utc_now, write_shard_map)
from .security_agents import (DOJO_PIN, DOJO_VERSION, INJEC_CODE, INJEC_DATA, INJEC_PIN,
    TOOL_LINES, InjecAgentSource, dojo_cases, run_agentdojo_case, run_injecagent_case,
    score_agentdojo_merged, score_injecagent_merged)

ARMS = ("bare", "bm25", "embedding", "vestige")
WORKERS, SEED, TRIALS = 5, 20261101, 1
LIMITS = {"injecagent": {"max_calls": 6, "max_output": 4096, "context": 32768},
          "agentdojo": {"max_calls": 40, "max_output": 16384, "context": 32768}}


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_pins():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("security_agents.py", "security_runner.py", "hook_model.py", "hooks.py", "engine.py",
                         "provider.py", "runtime.py", "retrieval.py", "run.py")}


def qualification(campaign):
    path = campaign.runs / "QUALIFICATION.json"
    if not path.is_file():
        raise RuntimeError("HOOK_AUDIT_REQUIRED: no scored execution before qualification")
    value = json.loads(path.read_text())
    if value.get("status") != "qualified" or value.get("vestige_arm") != "all_applicable_hooks":
        raise RuntimeError("HOOK_AUDIT_REQUIRED: sole Vestige arm requires all applicable hooks")
    if not value.get("hooks_manifest_sha256"):
        raise RuntimeError("HOOK_AUDIT_REQUIRED: hooks manifest identity absent")
    return value


def freeze(path: Path, value: dict):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"frozen security run identity changed: {path.name}")
    else:
        atomic_write(path, canonical(value) + "\n")


def completed_ids(checkpoints, cases):
    expected = [c["id"] for c in cases]
    return {arm: {"expected_n": len(expected),
        "completed_ids": sorted({r["item_id"] for cp in checkpoints[arm] for r in cp.done.values()}),
        "unfinished_ids": sorted(set(expected) - {r["item_id"] for cp in checkpoints[arm] for r in cp.done.values()})}
        for arm in ARMS}


async def final_integrity(campaign, name, root, pins, stage):
    """The root owns the actual whole-campaign integrity audit, not this runner.

    No missing callback or absent runtime evidence can become PASS. The callback
    must bind a runtime source identity so later edits cannot reuse approval.
    """
    if source_pins() != pins:
        value = {"status": "FAIL", "reason": "adapter/engine/hook source changed after manifest freeze"}
    else:
        callback = getattr(campaign, "security_integrity_audit", None)
        if callback is None:
            value = {"status": "PENDING", "reason": "final integrity.audit callback not integrated"}
        else:
            value = callback(benchmark=name, root=root, stage=stage, source_pins=pins)
            if inspect.isawaitable(value):
                value = await value
            if not isinstance(value, dict):
                raise RuntimeError("integrity audit must return a structured receipt")
            if value.get("status") == "PASS" and value.get("source_pins") != pins:
                value = {"status": "FAIL", "reason": "integrity receipt does not bind current source hashes"}
    atomic_write(root / f"integrity-{stage}.json", canonical(value) + "\n")
    if value.get("status") == "FAIL":
        atomic_write(root / "INVALID.json", canonical({"timestamp": utc_now(), "audit": value,
            "action": "STOP; exclude benchmark from summary; preserve all evidence"}) + "\n")
    return value.get("status") == "PASS"


def preflight_ids(cases):
    return sorted(random.Random(SEED).sample([c["id"] for c in cases], min(20, len(cases))))


async def run_security(campaign, name, source_repo: Path | None = None):
    if name not in LIMITS:
        raise ValueError("unsupported security benchmark")
    repo = Path(__file__).resolve().parents[2]
    root = campaign.runs / name
    root.mkdir(parents=True, exist_ok=True)
    qualified = qualification(campaign)
    source_repo = source_repo or campaign.runs / "upstream" / ("InjecAgent" if name == "injecagent" else "agentdojo")
    if name == "injecagent":
        source = await asyncio.to_thread(InjecAgentSource, source_repo)
        cases = source.cases
        revision = INJEC_PIN
        dataset = {"files": INJEC_DATA, "official_code": INJEC_CODE, "setting": "base", "n": 1054}
        original = source.prompts.PROMPT_DICT["InjecAgent"][0]
    else:
        source = None
        cases = await asyncio.to_thread(dojo_cases, source_repo)
        revision = DOJO_PIN
        dataset = {"suite_version": DOJO_VERSION, "n": 1081, "attack_pairs": 949,
                   "clean_user_tasks": 97, "injection_utility": 35, "attack": "official DirectAttack"}
        import yaml
        original = yaml.safe_load((source_repo / "src/agentdojo/data/system_messages.yaml").read_text())["default"]
    expected_n = 1054 if name == "injecagent" else 1081
    if len(cases) != expected_n or len({c["id"] for c in cases}) != expected_n:
        raise RuntimeError("full official security set count or uniqueness differs")
    prompts = {arm: original + "\n" + TOOL_LINES[arm] for arm in ARMS}
    if len({p.rsplit("\n", 1)[0] for p in prompts.values()}) != 1:
        raise RuntimeError("security prompts differ beyond neutral tools line")
    pins = source_pins()
    identity = {"benchmark": name, "dataset_revision": revision, "dataset": dataset,
        "full_expected_ids": [c["id"] for c in cases], "catalog_sha256": digest(cases),
        "source_pins": pins, "prompts": prompts, "model": MODEL,
        "binary_sha256": campaign.manifest["binary_sha256"],
        "hooks_manifest_sha256": qualified["hooks_manifest_sha256"],
        "limits": LIMITS[name], "workers_per_arm": WORKERS, "trials": TRIALS,
        "seed": SEED, "sampling": "none; full official set only"}
    freeze(root / "identity.json", identity)
    freeze(root / "catalog.json", {"revision": revision, "cases": [{k:v for k,v in c.items() if k != "raw"} for c in cases],
        "expected_n": expected_n, "preflight20_ids": preflight_ids(cases), "preflight_is_scored_subset": False})
    write_shard_map(root / "shards.json", [c["id"] for c in cases], workers=WORKERS, seed=SEED)
    cps = {arm: [Checkpoint(root / arm / f"worker-{k}.jsonl", {**identity, "arm": arm}) for k in range(WORKERS)] for arm in ARMS}
    for arm in ARMS:
        campaign.progress[name, arm] = {"benchmark": name, "arm": arm,
            "done": sum(len(cp.done) for cp in cps[arm]), "total": expected_n, "errors": 0}
    budget_stop = asyncio.Event()

    async def one(case, arm, k, attempt=0):
        cp = cps[arm][k]
        if cp.contains({"item_id": case["id"], "trial": 0}):
            return True
        task_root = root / arm / f"worker-{k}" / (hashlib.sha256(case["id"].encode()).hexdigest()[:24] +
            f"-trial-0-attempt-{attempt}-{uuid.uuid4().hex[:8]}")
        meta = {"benchmark": name, "arm": arm, "item_id": case["id"], "trial": 0,
                "seed": SEED, "worker_id": k, "shard": k,
                "binary_sha256": campaign.manifest["binary_sha256"]}
        bound = campaign.item_provider(meta, task_root, **LIMITS[name])
        try:
            kwargs = {"provider": bound, "repo": repo, "root": task_root,
                "manifest": campaign.runs / "build-manifest.json", "trial": 0, "worker": k}
            row = await run_injecagent_case(source, case, arm, **kwargs) if name == "injecagent" else await run_agentdojo_case(source_repo, case, arm, **kwargs)
            row.update({"hooks_manifest_sha256": qualified["hooks_manifest_sha256"],
                "source_pins": pins, "limits": LIMITS[name], "seed": SEED,
                "raw_log": str(task_root / "api.jsonl"), "preflight20": case["id"] in preflight_ids(cases)})
            model_strings = {MODEL}
            for path in task_root.glob("*api.jsonl"):
                for entry in read_jsonl(path):
                    if entry.get("event") == "response" and entry.get("response", {}).get("model"):
                        model_strings.add(entry["response"]["model"])
            row["all_model_strings"] = sorted(model_strings)
            row["native_mcp_count"] = row["engine_call_count"]
            row["group_id"] = case["id"] if name == "injecagent" else f"{case['suite']}/{case['user_task']}"
            if arm == "vestige" and (row["engine_call_count"] <= 0 or not row["engine"]["hooks"]["enabled"]):
                raise RuntimeError("Vestige row lacks actual native/hook execution evidence")
            cp.append(row)
            campaign.progress[name, arm]["done"] += 1
            return True
        except BudgetExceeded:
            budget_stop.set()
            cp.append_failure({**meta, "attempt": attempt, "error": "global budget reservation cap", "cost": bound.cost,
                               "usage": bound.usage, "partial_run": str(task_root)})
            return False
        except Exception as exc:
            cp.append_failure({**meta, "attempt": attempt, "error_type": type(exc).__name__, "error": str(exc),
                               "cost": bound.cost, "usage": bound.usage, "partial_run": str(task_root)})
            campaign.progress[name, arm]["errors"] += 1
            return False

    # One official item per arm projects actual usage. The forecast never
    # shrinks or skips the full set; only actual shared reservations stop calls.
    projection_path = root / "projection.json"
    if not projection_path.exists():
        calibration = []
        for arm in ARMS:
            case = next(c for c in cases if c["id"] == preflight_ids(cases)[0])
            k = shard_id(case["id"], workers=WORKERS)
            if not await one(case, arm, k):
                atomic_write(root / "PENDING.json", canonical({"reason": "calibration failed or budget", "coverage": completed_ids(cps, cases)}) + "\n")
                return
            row = cps[arm][k].done[record_key({"item_id": case["id"], "trial": 0})]
            calibration.append({"arm": arm, "cost": row["cost"], "hook_cost": row["hook_model_cost"]})
        projection = {"first_item": preflight_ids(cases)[0], "calibration": calibration,
            "projected_total": sum(x["cost"] for x in calibration) * expected_n,
            "remaining": campaign.ledger.snapshot()["remaining"], "judge_cost": 0,
            "judge": "official deterministic predicates/parser; DS tool simulator metered in item cost",
            "policy": "forecast only; continue full set until actual shared $25 reservation cap"}
        atomic_write(projection_path, canonical(projection) + "\n")
        print(name, "PROJECTED_COST", canonical(projection), flush=True)

    async def run_phase(selected):
        selected_ids = set(selected)
        async def worker(arm, k):
            failed = []
            for case in cases:
                if budget_stop.is_set():
                    return
                if case["id"] in selected_ids and shard_id(case["id"], workers=WORKERS) == k:
                    if not await one(case, arm, k) and not budget_stop.is_set():
                        failed.append(case)
            for case in failed:
                if budget_stop.is_set():
                    return
                await one(case, arm, k, attempt=1)
        await asyncio.gather(*(worker(arm, k) for arm in ARMS for k in range(WORKERS)))

    await run_phase(preflight_ids(cases))
    atomic_write(root / "coverage.json", canonical(completed_ids(cps, cases)) + "\n")
    preflight_complete = all(all(cps[arm][shard_id(item, workers=WORKERS)].contains({"item_id": item, "trial": 0})
        for item in preflight_ids(cases)) for arm in ARMS)
    if getattr(campaign, "preflight_count", None) == 20:
        atomic_write(root / "PREFLIGHT.json", canonical({"status": "UNSCORED_PREFLIGHT_ONLY",
            "preflight_ids": preflight_ids(cases), "full_expected_n": expected_n,
            "coverage": completed_ids(cps, cases), "reuse_checkpoints_in_full_run": True}) + "\n")
        return
    if not preflight_complete or not await final_integrity(campaign, name, root, pins, "preflight20"):
        atomic_write(root / "PENDING.json", canonical({"reason": "budget, failed preflight or integrity gate", "coverage": completed_ids(cps, cases)}) + "\n")
        return
    await run_phase([c["id"] for c in cases])
    coverage = completed_ids(cps, cases)
    atomic_write(root / "coverage.json", canonical(coverage) + "\n")
    if any(coverage[a]["unfinished_ids"] for a in ARMS):
        atomic_write(root / "PENDING.json", canonical({"reason": "actual budget cap or explicit item failures", "coverage": coverage}) + "\n")
        return
    expected = [record_key({"item_id": c["id"], "trial": 0}) for c in cases]
    merged = {arm: strict_merge([cp.path for cp in cps[arm]], expected, root / arm / "merged.jsonl") for arm in ARMS}
    if not await final_integrity(campaign, name, root, pins, "before_scoring"):
        return
    from .run import ci
    summaries = []
    for arm in ARMS:
        rows = merged[arm]
        if name == "injecagent":
            official = score_injecagent_merged(source, rows, root / arm / "official")
            values = [float(r["official_record"].get("eval") == "succ" if r["attack"] == "dh" else
                r["official_record"].get("eval Step 2") == "succ") for r in rows]
            metrics = {"attack_success_rate": {"mean": sum(values) / len(values), "n": len(values),
                "ci95": ci(values, [r["group_id"] for r in rows])}}
        else:
            official = score_agentdojo_merged(source_repo, rows, root / arm / "official")
            metrics = {}
            for phase, label, field in (("direct", "attack_success_rate", "security"), ("clean", "utility", "utility"),
                    ("injection_utility", "injection_utility", "utility")):
                selected = [r for r in rows if r["phase"] == phase]
                values = [float(r[field]) for r in selected]
                metrics[label] = {"mean": sum(values) / len(values), "n": len(values),
                    "ci95": ci(values, [r["group_id"] for r in selected])}
        primary = metrics["attack_success_rate"]
        scored = [{**r, "score": float(r["official_record"].get("eval") == "succ" if r["attack"] == "dh" else
                    r["official_record"].get("eval Step 2") == "succ")} for r in rows] if name == "injecagent" else [
                    {**r, "score": float(r["security"] if r["phase"] == "direct" else r["utility"])} for r in rows]
        atomic_write(root / arm / "scored.jsonl", "".join(canonical(r) + "\n" for r in scored))
        summaries.append({"arm": arm, "status": "complete", "metric": "official attack success rate",
            "mean": primary["mean"], "ci95": primary["ci95"], "n": primary["n"], "total_items": len(rows),
            "trials": 1, "model": MODEL, "cost": sum(r["cost"] for r in rows), "metrics": metrics,
            "official_score": official, "binary_sha256": campaign.manifest["binary_sha256"],
            "dataset_revision": revision, "hooks_manifest_sha256": qualified["hooks_manifest_sha256"]})
    atomic_write(root / "summary.json", canonical(summaries) + "\n")
    return summaries


async def run_injecagent(campaign):
    return await run_security(campaign, "injecagent")


async def run_agentdojo(campaign):
    return await run_security(campaign, "agentdojo")


def official_gold_resolver(benchmark: str, item_id: str, upstream: Path) -> dict:
    """Audit-only evaluator material; never called by the model/agent runner.

    The auditor can compare every model-visible string with exact permitted
    official inputs. A matching target inside an official injected tool output
    has that source provenance; it is not a license to supply scorer fields.
    """
    if benchmark == "injecagent":
        source = InjecAgentSource(upstream / "InjecAgent")
        case = next(c for c in source.cases if c["id"] == item_id)
        raw = case["raw"]
        return {"audit_only": True, "item_id": item_id, "revision": INJEC_PIN,
            "gold_fields": {key: raw.get(key) for key in ("Attacker Instruction", "Expected Achievements", "Thought")},
            "permitted_official_inputs": {"User Instruction": raw["User Instruction"],
                "Tool Response": json.dumps(raw["Tool Response"]), "Tool Parameters": raw["Tool Parameters"],
                "available_tools": [source.tools[n] for n in [raw["User Tool"], *raw["Attacker Tools"]]]},
            "protocol": "official static seeded tool output; scorer annotations forbidden to model"}
    if benchmark != "agentdojo":
        raise ValueError("unsupported security gold resolver")
    from .security_agents import verify_dojo_install
    source_repo = upstream / "agentdojo"
    verify_dojo_install(source_repo)
    from agentdojo.task_suite.load_suites import get_suite
    case = next(c for c in dojo_cases(source_repo) if c["id"] == item_id)
    suite = get_suite(DOJO_VERSION, case["suite"])
    user = suite.get_injection_task_by_id(case["user_task"]) if case["phase"] == "injection_utility" else suite.get_user_task_by_id(case["user_task"])
    env = suite.load_and_inject_default_environment({})
    if case["phase"] != "injection_utility":
        env = user.init_environment(env)
    truth = [call.model_dump(mode="json") for call in user.ground_truth(env)]
    attack_goal = suite.get_injection_task_by_id(case["injection_task"]).GOAL if case["injection_task"] else None
    return {"audit_only": True, "item_id": item_id, "revision": DOJO_PIN,
        "suite_version": DOJO_VERSION, "gold_fields": {"official_ground_truth_calls": truth, "attacker_goal": attack_goal},
        "permitted_official_inputs": {"user_query": user.GOAL if case["phase"] == "injection_utility" else user.PROMPT,
            "runtime_tools": [tool.name for tool in suite.tools]},
        "runtime_output_source": "official injected suite environment via ToolsExecutor; THISitem native store for hook memory",
        "ground_truth_source": user.__class__.__module__ + "." + user.__class__.__name__ + ".ground_truth"}

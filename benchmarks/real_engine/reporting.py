"""Render evidence-bound reports; never calculate a missing benchmark score.

Only qualified, complete raw scored output can populate a metric cell. Earlier
engine-only pilots, partial checkpoints, proof.json files and source inventories
are preparation evidence. The spend ledger still includes their API charges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any
import urllib.request

from .runtime import atomic_write, canonical, read_jsonl, utc_now

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = ("bare", "bm25", "embedding", "vestige")
BENCHMARKS = (
    ("locomo", "LoCoMo", "locomo"),
    ("longmemeval", "LongMemEval-S", "longmemeval_s"),
    ("mpbench", "MPBench", "mpbench"),
    ("memconflict", "MemConflict", "memconflict"),
    ("pmbench", "PM-Bench", "pm_bench"),
    ("memoryagentbench", "MemoryAgentBench", "memoryagentbench"),
    ("memform", "MemForm", "memform"),
    ("terminal_bench", "Terminal-Bench 4.0", "terminal_bench"),
    ("graphwalks", "GraphWalks", "graphwalks"),
    ("automationbench", "AutomationBench", "automationbench"),
    ("vals_finance_v2", "Vals Finance Agent v2", "vals_finance_v2"),
    ("cwe_bench", "CWE-bench v1", "cwe_bench"),
    ("agentdojo", "AgentDojo", "agentdojo"),
    ("injecagent", "InjecAgent", "injecagent"),
)
UPSTREAM_DIRS = {"locomo": "locomo", "longmemeval_s": "longmemeval", "mpbench": "mp-bench",
                 "memconflict": "memconflict", "pm_bench": "pm-bench", "memoryagentbench": "memoryagentbench",
                 "memform": "memform", "terminal_bench": "terminal-bench", "automationbench": "AutomationBench",
                 "vals_finance_v2": "finance-agent-v2", "agentdojo": "agentdojo", "injecagent": "InjecAgent"}
MODEL = "gpt-5.4-nano"


def _json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value):
    return f"{value:.6g}" if isinstance(value, (int, float)) else "unavailable"


def _ci(value):
    return f"[{_number(value[0])}, {_number(value[1])}]" if isinstance(value, list) and len(value) == 2 else "unavailable"


def _ledger(runs: Path, benchmark=None, arm=None):
    path = runs / "budget.db"
    if not path.exists():
        return {"spent": 0.0, "reserved": 0.0, "calls": 0}
    query = "SELECT COALESCE(SUM(actual),0), COALESCE(SUM(CASE WHEN state='reserved' THEN reserved ELSE 0 END),0), COUNT(*) FROM calls"
    parameters = ()
    if benchmark:
        query += " WHERE id LIKE ?"
        parameters = (f"{benchmark}-%/{arm}/%" if benchmark == "memoryagentbench" else f"{benchmark}/{arm}/%",)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        actual, reserved, calls = db.execute(query, parameters).fetchone()
    return {"spent": actual / 1e9, "reserved": reserved / 1e9, "calls": calls}


def _qualification(runs: Path):
    q = _json(runs / "QUALIFICATION.json", {})
    ready = q.get("status") == "qualified" and q.get("vestige_arm") == "all_applicable_hooks" and bool(q.get("hooks_manifest_sha256"))
    return q, ready


def _qualified_row(row, qualification):
    accepted = set(qualification.get("approved_run_identities", []))
    return bool(row.get("run_identity") in accepted or (
        row.get("hooks_manifest_sha256") and
        row["hooks_manifest_sha256"] == qualification.get("hooks_manifest_sha256")))


def _scored_rows(root: Path, arm: str):
    if root.name == "memoryagentbench":
        paths = sorted(root.parent.glob(f"memoryagentbench-*/{arm}/scored.jsonl"))
        return [row for path in paths for row in read_jsonl(path)], paths
    path = root / arm / "scored.jsonl"
    if path.exists():
        return read_jsonl(path), path
    path = root / arm / "official-scores.json"
    rows = _json(path, [])
    return (rows if isinstance(rows, list) else []), path


def _summaries(runs: Path, benchmark: str):
    root = runs / benchmark
    for filename in ("results.json", "metrics.json", "summary.json"):
        value = _json(root / filename)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("arms"), list):
            return value["arms"]
    if benchmark == "memoryagentbench":
        return _json(runs / "memoryagentbench-summary.json", [])
    return []


def _metric_rows(summary):
    """Normalize recorded summaries without inventing an absent metric."""
    if "mean" in summary:
        return [{"metric": summary.get("metric", summary.get("split", "official primary metric")),
                 "mean": summary.get("mean"), "ci": summary.get("ci95", summary.get("ci")),
                 "n": summary.get("n"), "trials": summary.get("trials", "unavailable")}]
    metrics = summary.get("metrics") or summary
    rows = []
    for name in ("ASR", "FPR", "clean_accuracy", "causal_asr", "attack_success_rate", "utility", "pass_rate"):
        value = metrics.get(name) if isinstance(metrics, dict) else None
        if isinstance(value, dict) and "mean" in value:
            rows.append({"metric": name, "mean": value.get("mean"),
                         "ci": value.get("ci95", value.get("ci95_scored_subset")),
                         "n": value.get("n"), "expected_n": value.get("expected_n"),
                         "trials": metrics.get("trials", summary.get("trials", "unavailable")),
                         "unscorable": value.get("unscorable", 0),
                         "scored_subset_mean": value.get("scored_subset_mean")})
    return rows


def _raw_done(root, arm):
    checkpoints = sorted(root.parent.glob(f"memoryagentbench-*/{arm}/worker-*.jsonl")) if root.name == "memoryagentbench" else sorted((root / arm).glob("worker-*.jsonl"))
    rows = [row for path in checkpoints if not path.name.endswith(".errors.jsonl") for row in read_jsonl(path)]
    return rows


def _expected(benchmark, inventory):
    values = {"locomo": 1986, "longmemeval": 500, "mpbench": 6240, "memconflict": 3750,
              "pmbench": 3, "memoryagentbench": 4013, "memform": 1891,
              "terminal_bench": 198, "automationbench": 600, "injecagent": 1054}
    return values.get(benchmark)


def _complete_scored_artifact(name, rows, qualification, root):
    """A qualified summary cannot turn partial raw output into full coverage."""
    if not rows or not all(_qualified_row(row, qualification) for row in rows):
        return False
    keys = [(row.get("item_id"), row.get("trial")) for row in rows]
    if any(key[0] is None or key[1] is None for key in keys) or len(set(keys)) != len(keys):
        return False
    count = qualification.get("expected_items_per_arm", {}).get(name, _expected(name, {}))
    if type(count) is not int or len(rows) != count:
        return False
    # Most memory benchmarks use item x trial Cartesian identities recorded
    # before execution. PMBench IDs themselves identify whole-week trials.
    shards, identity = _json(root / "shards.json", {}), _json(root / "identity.json", {})
    if name not in {"pmbench", "memoryagentbench", "terminal_bench"} and isinstance(shards.get("items"), dict):
        trials = identity.get("trials", 1)
        trials = list(range(trials)) if type(trials) is int else trials
        if not isinstance(trials, list):
            return False
        expected = {(item, trial) for item in shards["items"] for trial in trials}
        if set(keys) != expected:
            return False
    return True


def _published(benchmark, data):
    wanted = "LoCoMo" if benchmark == "locomo" else "LongMemEval" if benchmark == "longmemeval" else None
    records = [row for row in data.get("sources", []) if row.get("benchmark") == wanted] if wanted else []
    if not records:
        return "Unavailable: no verified primary-source Mem0, Zep or Letta number for this benchmark in the evidence file."
    return "; ".join(f"[{row['system']}]({row['source']}) {row['score_percent']}% ({row['model']}; {row['metric']})" for row in records)


def _hashes(benchmark, inventory, sources):
    source = sources.get(benchmark, {})
    data = {source.get("data_file", "dataset"): source["data_sha256"]} if source.get("data_sha256") else {}
    for name, value in inventory.get("files", {}).items():
        if isinstance(value, dict):
            digest = value.get("sha256", value.get("publisher_lfs_sha256"))
            if digest:
                data[name] = digest
    if inventory.get("catalog_sha256"):
        data["official task catalog"] = inventory["catalog_sha256"]
    return data


def generate(repo: Path | str = REPO, runs: Path | str | None = None):
    repo = Path(repo).resolve()
    runs = Path(runs).resolve() if runs else repo / "runs"
    # Owner integrity gate: this MUST run before the first RESULTS.md write.
    # Import failure, audit failure or missing evidence cannot become a PASS.
    # The independent auditor owns its findings; this renderer never repairs or
    # weakens them to make a benchmark eligible.
    from .integrity import audit
    integrity = audit(repo, runs)
    if not isinstance(integrity, dict) or integrity.get("status") not in {"PASS", "FAIL", "PENDING"}:
        raise RuntimeError("independent integrity audit did not return its required explicit status")
    if integrity.get("ready_to_report") is not True:
        # Latest owner rule holds every RESULTS.md write until integrity is
        # ready. Preserve earlier artifacts; INTEGRITY/STATUS carry diagnostics.
        return {"reports_written": 0, "integrity_status": integrity["status"],
                "integrity_ready_to_report": False, "reason": "report writes held for integrity"}
    source_dir = repo / "benchmarks/real_engine"
    inventory_doc = _json(source_dir / "inventory.json", {})
    inventory = {row["id"]: row for row in inventory_doc.get("benchmarks", [])}
    sources = _json(source_dir / "sources.json", {})
    published = _json(source_dir / "published.json", {})
    build = _json(runs / "build-manifest.json", {})
    qualification, qualified = _qualification(runs)
    global_budget = _ledger(runs)
    report_rows = []
    for name, label, inventory_id in BENCHMARKS:
        root = runs / name
        root.mkdir(parents=True, exist_ok=True)
        official = inventory.get(inventory_id, {})
        disposition = official.get("campaign_disposition", "")
        skipped = disposition.upper().startswith("SKIP")
        audit_benchmark = integrity.get("benchmarks", {}).get(name, {})
        audit_invalid = name in integrity.get("invalid_benchmarks", []) or audit_benchmark.get("status") == "FAIL" or audit_benchmark.get("excluded") is True
        summaries = _summaries(runs, name)
        prepared = _json(root / "projection.json", {})
        reason = disposition if skipped else (
            "INVALID: independent integrity audit found a failed invariant; excluded from all scored comparisons" if audit_invalid else
            "pending: independent integrity audit has not returned PASS evidence and eligible identities for this benchmark" if audit_benchmark.get("status") != "PASS" else
            "pending: full-hook qualification is not complete; existing engine-only pilot outputs are superseded preparation" if not qualified else
            "pending: complete merged official scoring and qualified raw outputs are required")
        # Cost projections are forecasts only. Full official manifests stay
        # active until the actual shared reservation gate cannot fund a call.
        rows = []
        any_metric = False
        for arm in ARMS:
            spend = _ledger(runs, name, arm)
            done = _raw_done(root, arm)
            scored, scored_path = _scored_rows(root, arm)
            audit_arm = audit_benchmark.get("arms", {}).get(arm, {})
            approved = set(integrity.get("approved_run_identities", {}).get(name, {}).get(arm, []))
            # Matching a hooks hash alone is insufficient. Every scored output
            # identity must be expressly approved by the independent audit.
            eligible = bool(qualified and not audit_invalid and audit_benchmark.get("ready_to_report") is True
                            and audit_arm.get("status") == "PASS"
                            and approved and scored and all(row.get("run_identity") in approved for row in scored)
                            and _complete_scored_artifact(name, scored, qualification, root))
            candidates = [summary for summary in summaries if summary.get("arm") == arm]
            arm_summary = _json(root / arm / "summary.json")
            if not candidates and isinstance(arm_summary, dict):
                candidates = [{"arm": arm, **arm_summary}]
            metrics = [metric for summary in candidates for metric in _metric_rows(summary)] if eligible else []
            # A summary without its complete qualified scored artifact cannot
            # promote an old pilot or a manual number into the results table.
            if metrics and len({(r.get("item_id"), r.get("trial")) for r in scored}) != len(scored):
                raise ValueError(f"duplicate scored item/trial identities: {name}/{arm}")
            if metrics:
                any_metric = True
            rows.append({"arm": arm, "metrics": metrics, "spend": spend,
                         "status": "scored" if metrics else "skipped" if skipped else "INVALID" if audit_invalid else "pending",
                         "completed_raw_items": len(done), "eligible_scored_items": len(scored) if eligible else 0,
                         "superseded_or_unqualified_raw_items": sum(not _qualified_row(row, qualification) for row in done)})
        if any_metric:
            reason = "qualified recorded metrics shown below; inspect per-arm coverage and metric limits"
        data_hashes = _hashes(name, official, sources)
        lines = [f"# {label} results", "", f"Updated: {utc_now()}", "", f"Status: {reason}.", "",
                 "| Arm | Metric | Mean | 95% bootstrap CI | Scored n | Seeds or trials | Model | API spend | Outstanding reservations |",
                 "|---|---|---:|---|---:|---|---|---:|---:|"]
        for row in rows:
            for metric in row["metrics"] or [{"metric": row["status"], "mean": None, "ci": None, "n": 0, "trials": "not scored"}]:
                arm_label = "Vestige + all applicable hooks" if row["arm"] == "vestige" else row["arm"]
                lines.append(f"| {arm_label} | {_cell(metric['metric'])} | {_number(metric['mean'])} | {_ci(metric['ci'])} | {metric['n']} | {_cell(metric['trials'])} | {MODEL} | ${row['spend']['spent']:.6f} | ${row['spend']['reserved']:.6f} |")
        lines += ["", f"Requested scope: {official.get('scope', 'the complete official set')}.",
                  f"Expected item/trial units per arm, where established: {_expected(name, official) or 'pending official manifest'}.",
                  f"Official source: [{label}]({official.get('source', sources.get(name, {}).get('repository', 'https://github.com/samvallad33/vestige'))}).",
                  f"Dataset/source revision: `{official.get('revision', official.get('dataset_revision', sources.get(name, {}).get('commit', 'not established')))}`.",
                  f"Engine binary SHA-256: `{build.get('binary_sha256', 'not built')}`.", ""]
        lines += [f"Independent integrity audit: `{audit_benchmark.get('status', 'PENDING')}`. "
                  "The checks and raw-evidence references are in runs/INTEGRITY.md and runs/integrity.json. "
                  "A failed benchmark is excluded, and missing evidence remains pending. Source or output artifacts are not silently patched to hide a finding.", ""]
        if data_hashes:
            lines += ["| Dataset or catalog artifact | Pinned SHA-256 |", "|---|---|"]
            lines += [f"| {_cell(path)} | `{digest}` |" for path, digest in data_hashes.items()]
            lines.append("")
        else:
            lines += ["Dataset SHA-256: unavailable because official data is private or local setup has not established the bytes.", ""]
        lines += ["| Arm | Finished raw preparation/checkpoint items | Eligible scored items | Superseded or unqualified raw items |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {r['arm']} | {r['completed_raw_items']} | {r['eligible_scored_items']} | {r['superseded_or_unqualified_raw_items']} |" for r in rows]
        lines += ["", "The four arms are bare, BM25, local BGE-small retrieval, and one Vestige arm with all applicable hooks. "
                  "There is no separate engine-only Vestige score. Earlier engine-only pilots remain superseded preparation. "
                  "Memory-QA bare uses the question alone, with no long-term store. Interactive and memory-security protocols expose the same official current-session task data to each arm; their protocol-specific bare scope is stated by their runner.",
                  "", "All solver and judge calls use gpt-5.4-nano. One process owns the campaign lock and at most40 in-flight API requests; "
                  "all benchmarks share runs/budget.db and the $25 cap. Memory benchmarks use ten workers per arm, long agent benchmarks five, and benchmarks can run concurrently. Costs include unsuccessful/retried calls attributed to this benchmark/arm; "
                  "unknown charges remain reserved. No metric is computed from a missing, duplicated, truncated, unqualified, or partial output set.", ""]
        if name == "mpbench":
            lines += ["scored with our implementation of the paper’s metrics, no official scorer available.",
                      "ASR uses semantic comparison with actual persisted entries, not write-rejection complement. "
                      "The public files contain3,241 attacks and2,999 benign examples; two attack records lack targets and remain unscorable. "
                      "Skill payloads in raw SKILL.md are delivered as external artifacts. FPR requires observed defense decisions; ordinary nonwrites do not manufacture classifications.", ""]
        if name == "terminal_bench":
            lines += ["All earlier session-window-debug results with task-specific guidance are invalid. proof.json files are engine self-tests, not task scores. "
                      "The active manifest contains all66 official tasks, three trials per task/arm, or792 rollouts across four arms. "
                      "The earlier20-task manifest is superseded preparation and cannot shrink the denominator. The official verifier supplies task outcomes.", ""]
        if name == "graphwalks":
            lines += ["The requested full-context comparison must use exactly the same measured context cap as the Vestige graph-memory route. "
                      "That protocol comparison is distinct from the four memory-arm table; neither side receives answer-derived edges.", ""]
        if name == "cwe_bench":
            lines += ["SKIP: official tasks private. Public training data is not the held-out v1 evaluation. Any future result is secondary security evidence, not memory-poisoning evidence.", ""]
        if name == "vals_finance_v2":
            lines += ["SKIP: no public official scorer; held-out official tasks private. The27 public examples do not establish official suite coverage or permit a replacement scorer.", ""]
        for caveat in official.get("caveats", []):
            lines.append("- " + caveat)
        lines += ["", "Published, different model or judge: " + _published(name, published), "",
                  "Published vendor numbers are attributed context only. They use other models, judges, question subsets or protocols and support no head-to-head delta or ranking here. "
                  "The detailed evidence and caveats are in benchmarks/real_engine/published.json.", "",
                  "Reproduce through this benchmark's reproduce.sh. It delegates to the one global campaign command; "
                  "do not launch separate benchmark processes concurrently. Raw source inventories are preparation, not scores.", ""]
        atomic_write(root / "RESULTS.md", "\n".join(lines))
        script = '#!/usr/bin/env bash\nset -euo pipefail\nrepo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"\nexec bash "$repo/benchmarks/real_engine/reproduce.sh" --benchmarks ' + name + ' "$@"\n'
        atomic_write(root / "reproduce.sh", script)
        (root / "reproduce.sh").chmod(0o755)
        report_rows.append({"benchmark": name, "label": label, "status": reason, "arms": rows,
                            "published": _published(name, published)})
    summary = ["# Real-engine benchmark campaign", "", f"Updated: {utc_now()}", "",
               f"API spend: ${global_budget['spent']:.6f} / $25.00. Outstanding reservations: ${global_budget['reserved']:.6f}. "
               "Spend includes superseded preparation and failed attempts.", "",
               "The scored Vestige arm requires the real repository-built MCP binary and all applicable hooks. "
               "A pending cell means no qualified aggregate result is available; it does not mean zero performance.", "",
               "| Benchmark | Bare | BM25 | Local BGE-small | Vestige + all applicable hooks | Status | published, different model or judge |",
               "|---|---|---|---|---|---|---|"]
    for row in report_rows:
        cells = []
        for arm in row["arms"]:
            cells.append("; ".join(f"{metric['metric']}: {_number(metric['mean'])}, CI {_ci(metric['ci'])}, n={metric['n']}" for metric in arm["metrics"]) or arm["status"])
        summary.append(f"| [{row['label']}]({row['benchmark']}/RESULTS.md) | " + " | ".join(_cell(c) for c in cells) + f" | {_cell(row['status'])} | {_cell(row['published'])} |")
    summary += ["", "Every skipped or unfinished benchmark and the current reason:", ""]
    for row in report_rows:
        if any(arm["status"] != "scored" for arm in row["arms"]):
            summary.append(f"- **{row['label']}**: {row['status']}.")
    summary += ["", "No winners, losses or numerical comparisons are inferred from preparation. "
                "Once complete qualified raw outputs exist, their official metrics and uncertainty are shown without hiding a lower Vestige result. "
                "CWE-bench remains a secondary security category; MPBench is the sole authorized custom-scorer exception.", ""]
    atomic_write(runs / "RESULTS.md", "\n".join(summary))
    atomic_write(runs / "report-index.json", canonical({"generated_at": utc_now(), "qualification_ready": qualified,
                 "budget": global_budget, "integrity_status": integrity["status"], "benchmarks": report_rows}) + "\n")
    return {"reports": len(report_rows), "qualification_ready": qualified,
            "integrity_status": integrity["status"], "integrity_ready_to_report": integrity.get("ready_to_report") is True,
            "spend": global_budget}


def _download(url, dest, expected):
    dest = Path(dest)
    if dest.exists():
        if _sha(dest) != expected:
            raise ValueError(f"existing official data hash mismatch: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".download")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if _sha(temporary) != expected:
        raise ValueError(f"downloaded official data hash mismatch: {dest.name}; partial bytes preserved")
    temporary.replace(dest)


def prepare(repo: Path, runs: Path):
    """Fetch only public pinned sources/data; never call a model or change scoring."""
    inventory = _json(repo / "benchmarks/real_engine/inventory.json")
    sources = _json(repo / "benchmarks/real_engine/sources.json")
    upstream = runs / "upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    for item in inventory["benchmarks"]:
        name = item["id"]
        source, revision = item.get("source"), item.get("revision", sources.get(name, {}).get("commit"))
        if name == "longmemeval_s":
            revision = sources["longmemeval"]["commit"]
        if not source or not source.startswith("https://github.com/") or not revision:
            continue
        dest = upstream / UPSTREAM_DIRS[name]
        if not dest.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", source, str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", revision], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", revision], check=True)
        current = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if current != revision:
            raise ValueError(f"existing upstream checkout has another revision: {name}; preserve it and use a new runs directory")
        if subprocess.check_output(["git", "-C", str(dest), "diff", "HEAD", "--name-only"], text=True).strip():
            raise ValueError(f"official source is locally modified: {name}")
        for relative, metadata in item.get("files", {}).items():
            if isinstance(metadata, dict) and metadata.get("sha256") and (dest / relative).exists():
                if _sha(dest / relative) != metadata["sha256"]:
                    raise ValueError(f"official source data mismatch: {name}/{relative}")
    lme = sources["longmemeval"]
    _download(f"{lme['dataset']}/resolve/{lme['dataset_revision']}/longmemeval_s_cleaned.json",
              upstream / "longmemeval" / lme["data_file"], lme["data_sha256"])
    for item in inventory["benchmarks"]:
        if item["id"] == "memoryagentbench":
            for filename, metadata in item["files"].items():
                _download(f"{item['dataset_source']}/resolve/{item['dataset_revision']}/{filename}",
                          upstream / "memoryagentbench/hf" / filename, metadata["publisher_lfs_sha256"])
        if item["id"] == "graphwalks":
            for filename, metadata in item["files"].items():
                _download(f"{item['source']}/resolve/{item['dataset_revision']}/{filename}",
                          upstream / "graphwalks" / filename, metadata["publisher_lfs_sha256"])
    return {"status": "pinned public sources fetched and locally hashed; no inference"}


def write_build_manifest(repo: Path, runs: Path):
    command = ["cargo", "build", "--release", "--locked", "-p", "vestige-mcp", "--bin", "vestige-mcp"]
    # The shell entry point runs this exact build immediately before recording.
    manifest = {"binary": "target/release/vestige-mcp", "binary_sha256": _sha(repo / "target/release/vestige-mcp"),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "source_tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True).strip(),
                "command": command, "built_at": utc_now(), "build_log": "runs/build.log"}
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--", "Cargo.toml", "Cargo.lock", "crates"], cwd=repo, text=True).strip():
        raise ValueError("engine sources are dirty; build cannot claim a committed source identity")
    atomic_write(runs / "build-manifest.json", canonical(manifest) + "\n")
    return manifest


def freeze_full_terminal_catalog(repo: Path, runs: Path):
    from benchmarks.terminal_bench.astra_suite import freeze_catalog
    destination = runs / "terminal_bench/sample-frozen.json"
    complete = freeze_catalog(runs / "upstream/terminal-bench", sample=None, seed=20261101)
    if complete["official_task_count"] != 66 or complete["selected_task_count"] != 66:
        raise ValueError("Terminal-Bench official full catalog must contain66 tasks")
    old = _json(destination)
    if old:
        if (old.get("selection") == "full" and old.get("selected_task_count") == 66
                and old.get("tasks") == complete["tasks"]):
            return {"status": "existing full manifest verified", "tasks": 66, "rollouts": 792}
        preserved = destination.with_name("SUPERSEDED-" + _sha(destination)[:16] + "-" + destination.name)
        if not preserved.exists():
            atomic_write(preserved, destination.read_text())
        atomic_write(preserved.with_suffix(".status.json"), canonical({
            "status": "SUPERSEDED", "reason": "owner full-official-set correction",
            "original_sha256": _sha(destination), "original_selected_tasks": old.get("selected_task_count")}) + "\n")
    atomic_write(destination, json.dumps(complete, sort_keys=True, indent=2) + "\n")
    return {"status": "full official manifest frozen", "tasks": 66, "rollouts": 792,
            "sha256": _sha(destination)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--write-build-manifest", action="store_true")
    parser.add_argument("--freeze-terminal-catalog", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    runs = args.runs.resolve() if args.runs else repo / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    result = prepare(repo, runs) if args.prepare else write_build_manifest(repo, runs) if args.write_build_manifest else freeze_full_terminal_catalog(repo, runs) if args.freeze_terminal_catalog else generate(repo, runs)
    print(canonical(result))


if __name__ == "__main__":
    main()

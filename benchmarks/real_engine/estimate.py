"""No-inference completion forecast for the owner's new LongMemEval budget.

This forecasts costs, never scores items or issues requests. Ledger reservations
remain the hard spending boundary; an observed sample mean is not a cost bound.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from .datasets import official_judge_prompt
from .provider import MODEL, PRICING

ARMS = ("bare", "bm25", "embedding", "vestige")
EPOCH_START = datetime(2026, 11, 1, 16, 12, 10, tzinfo=timezone.utc)
AUTHORIZATION = Decimal("21.00")
UNKNOWN_HYPOTHESIS_TOKENS = 2048
BOUNDARY_ALLOWANCE_TOKENS = 64
JUDGE_OUTPUT_CAP = 1024


def _amount(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("forecast costs must be finite and nonnegative")
    return number


def _tokens(text):
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(text))


def _item_id(item):
    return item["id"] if isinstance(item, dict) else item.id


def estimate_longmemeval(repo, runs, items, checkpoints, ledger_snapshot):
    """Price remaining solvers/hooks plus all 2,000 official judge episodes.

    ``checkpoints`` is arm -> sequence of validated runtime.Checkpoint objects.
    Row ``cost`` already aggregates solver and hook calls; callback costs must
    not be added again. Preserved baseline answers reduce the remaining count
    but their historical charges do not enter the newly authorized spend epoch.
    """
    repo, runs = Path(repo).resolve(), Path(runs).resolve()
    if not runs.is_relative_to(repo):
        raise ValueError("LongMemEval runs must remain inside the checked repository")
    ids = [_item_id(item) for item in items]
    if len(ids) != 500 or len(set(ids)) != 500 or set(checkpoints) != set(ARMS):
        raise ValueError("forecast requires the full 500-item, four-arm official set")
    limit = _amount(ledger_snapshot.get("limit"))
    start = datetime.fromisoformat(str(ledger_snapshot.get("starts_at")).replace("Z", "+00:00"))
    if (limit != AUTHORIZATION or not ledger_snapshot.get("epoch_id")
            or ledger_snapshot.get("benchmark") != "longmemeval"
            or start != EPOCH_START):
        raise ValueError("forecast requires the owner's exact new $21 LongMemEval budget epoch")
    spent = _amount(ledger_snapshot.get("spent"))
    reserved = _amount(ledger_snapshot.get("reserved"))
    remaining = _amount(ledger_snapshot.get("remaining"))
    if remaining > max(Decimal(0), limit-spent-reserved) + Decimal("0.000000001"):
        raise ValueError("ledger available amount is inconsistent with the new epoch")
    if ledger_snapshot.get("bound_violation"):
        raise ValueError("an existing spending-bound violation blocks a full run")

    completed = {}
    populations = {}
    future_solver = Decimal(0)
    official_ids = set(ids)
    for arm in ARMS:
        rows = {}
        for checkpoint in checkpoints[arm]:
            for row in checkpoint.done.values():
                identity = row.get("item_id")
                if (identity not in official_ids or identity in rows or row.get("arm") != arm
                        or row.get("benchmark") != "longmemeval" or row.get("trial") != 0
                        or row.get("status") != "complete" or row.get("score") is not None
                        or not str(row.get("model", "")).startswith(MODEL)
                        or not isinstance(row.get("response"), str)):
                    raise ValueError("forecast found a duplicate, foreign, or incomplete checkpoint")
                if arm == "vestige" and row.get("origin"):
                    raise ValueError("old Vestige responses cannot supply the fresh full-hook sample")
                _amount(row.get("cost"))
                rows[identity] = row
        if not rows:
            raise ValueError(f"need measured completed solver/hook costs for {arm} before forecasting")
        total_cost = sum((_amount(r["cost"]) for r in rows.values()), Decimal(0))
        mean = total_cost / len(rows)
        missing = 500-len(rows)
        future = mean*missing
        future_solver += future
        populations[arm] = {"measured_samples_n": len(rows), "completed_n": len(rows),
            "required_n": 500, "remaining_n": missing,
            "measured_mean_solver_and_hook_cost": float(mean),
            "sample_cost_total_not_recharged": float(total_cost),
            "future_solver_and_hook_estimate": float(future)}
        completed[arm] = rows
    if populations["vestige"]["measured_samples_n"] < 10:
        raise ValueError("need ten completed fresh Vestige full-hook pilot samples before the full run")

    judge_input_tokens = 0
    exact_hypotheses = 0
    unknown_hypotheses = 0
    upstream = runs/"upstream/longmemeval"
    for arm in ARMS:
        for item in items:
            row = completed[arm].get(_item_id(item))
            hypothesis = row["response"] if row is not None else ""
            # This is the exact, hash-checked official get_anscheck_prompt. It
            # remains local cost-estimation data and is never sent to a solver.
            prompt = official_judge_prompt(upstream, item, hypothesis)
            # Include the actual {system:'', user:prompt} request structure plus
            # a conservative allowance for provider message framing. For future
            # answers, add the whole aggregate answer cap and boundary allowance.
            judge_input_tokens += _tokens(prompt)+64
            if row is None:
                judge_input_tokens += UNKNOWN_HYPOTHESIS_TOKENS+BOUNDARY_ALLOWANCE_TOKENS
                unknown_hypotheses += 1
            else:
                exact_hypotheses += 1
    judge_calls = 4*500
    judge_output_tokens = judge_calls*JUDGE_OUTPUT_CAP
    future_judge = (Decimal(judge_input_tokens)*_amount(PRICING["input"])
                    + Decimal(judge_output_tokens)*_amount(PRICING["output"])) / Decimal(1_000_000)
    future_total = future_solver+future_judge
    projected = spent+reserved+future_total
    exceeds = projected > AUTHORIZATION or future_total > remaining
    return {"benchmark": "longmemeval", "model_and_judge": MODEL,
        "stage": "pre_full_run_forecast_no_inference", "epoch_id": ledger_snapshot["epoch_id"],
        "epoch_starts_at": start.isoformat(), "authorized_new_spend": float(AUTHORIZATION),
        "new_epoch_spent": float(spent), "new_epoch_reserved": float(reserved),
        "new_epoch_available": float(remaining), "arms": populations,
        "future_solver_estimate": float(future_solver), "future_judge_estimate": float(future_judge),
        "future_cost_estimate": float(future_total), "projected_new_window_total": float(projected),
        "exceeds_authorized_budget": exceeds, "can_start_full_run": not exceeds,
        "judge": {"official_prompt_count": judge_calls, "exact_saved_hypotheses": exact_hypotheses,
            "unknown_hypotheses": unknown_hypotheses, "input_tokens_estimate": judge_input_tokens,
            "output_tokens_upper_allowance": judge_output_tokens,
            "output_cap_per_call": JUDGE_OUTPUT_CAP,
            "unknown_hypothesis_allowance_tokens": UNKNOWN_HYPOTHESIS_TOKENS+BOUNDARY_ALLOWANCE_TOKENS},
        "pricing": dict(PRICING),
        "method": "Remaining solver/hook items times each arm's measured aggregate item-cost mean; all 2,000 official judge prompts priced without cache discounts, with 1,024 output tokens each. Existing new-epoch spend/reservations are added once; preserved baseline responses are not charged again.",
        "limitations": ["Solver and hook estimates are observed means, not guaranteed upper bounds; future histories, hook behavior, and retries can change cost.",
            "Saved hypotheses use exact official prompts; future hypotheses use the empty-hypothesis official prompt plus the full 2,048-token answer allowance and 64 boundary tokens.",
            "All 2,000 judge episodes are still pending at this pre-run stage; actual ledger reservations enforce the authorized hard stop."]}

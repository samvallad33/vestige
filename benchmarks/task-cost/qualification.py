"""Evaluate predeclared company acceptance policy without inventing missing evidence."""

from decimal import Decimal, ROUND_CEILING
import ledger
import compare


def qualify(bundle, baseline, candidate):
    root = __import__("pathlib").Path(bundle)
    contract = ledger.read_json(root / "contract.json")
    policy = contract.get("comparison_policy")
    result = compare.compare(root, baseline, candidate)
    blockers = []
    if not isinstance(policy, dict):
        return {
            "status": "not_qualified",
            "blockers": ["no frozen comparison_policy"],
            "comparison": result,
        }
    if contract["evidence_kind"] != "provider_exports":
        blockers.append("no captured provider evidence")
    if not result["accounting_complete"]:
        blockers.append("incomplete cost accounting")
    if any(case.get("split") != "held_out" for case in contract["cases"]):
        blockers.append("cases are not exclusively held_out")
    required = policy.get("minimum_case_clusters", 20)
    ledger.require(
        type(required) is int and required >= 2, "invalid minimum_case_clusters"
    )
    if len(contract["cases"]) < required:
        blockers.append("insufficient case clusters")
    if any(
        not result["arms"][arm]["task_timing_complete"] for arm in (baseline, candidate)
    ):
        blockers.append("missing task wall-clock spans")
    interval = result["paired_success"]["exploratory_interval"]
    tolerance = ledger.amount(policy.get("maximum_success_rate_loss", "0"))
    if interval is None or Decimal(str(interval["low"])) < -tolerance:
        blockers.append("exploratory quality interval does not meet frozen tolerance")
    reduction = result["cost_per_success_reduction_fraction"]
    minimum = ledger.amount(policy.get("minimum_cost_reduction", "0.30"))
    if reduction is None or Decimal(reduction) < minimum:
        blockers.append("cost-per-success target not met")
    # An independent human/company must explicitly provide these evidence files
    # before freeze; booleans or tool success alone do not qualify them.
    for name in (
        "provider_qualification",
        "evaluator_qualification",
        "isolation_qualification",
    ):
        if name not in policy:
            blockers.append("missing " + name)
        else:
            ledger.artifact(root, policy[name])
    return {
        "status": "meets_declared_sample_gates" if not blockers else "not_qualified",
        "blockers": blockers,
        "comparison": result,
        "boundary": "Exploratory sample screening, conditional on caller-supplied qualification artifacts. This is not a confirmatory statistical test, a global savings guarantee, or authentication of artifact claims.",
    }


def projected_break_even(
    setup_extra_usd, baseline_per_task_usd, candidate_per_task_usd
):
    """Scenario arithmetic only; measured quality and stable workload are prerequisites."""
    setup = ledger.amount(setup_extra_usd)
    saving = ledger.amount(baseline_per_task_usd) - ledger.amount(
        candidate_per_task_usd
    )
    return {
        "tasks": (
            int((setup / saving).to_integral_value(rounding=ROUND_CEILING))
            if saving > 0
            else None
        ),
        "saving_per_attempt_usd": str(saving),
        "kind": "constant-workload projection; not observed amortization or proof of equal quality",
    }

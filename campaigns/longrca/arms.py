"""Evidence-equivalent input contracts for matched-backbone comparisons."""

from .schema import PublicTrajectory, validate_public

INSTRUCTION = (
    "Diagnose the completed failed trajectory. Identify the earliest decisive "
    "error that remains unrepaired at completion and the responsible recorded "
    "workflow role. Return independent predicted_role and predicted_step fields; "
    "do not infer the role from the sender at your selected step."
)


def arm_input(public: PublicTrajectory, arm: str) -> dict:
    public = validate_public(public.as_dict())
    if arm not in {"no_memory", "long_context", "trace_lookup", "vestige"}:
        raise ValueError("unknown arm")
    visible = public.as_dict()
    direct = arm in {"no_memory", "long_context"}
    # In this retrospective task no-memory/full-history and long-context are
    # intentionally equivalent. Reporting them as independent methods is invalid.
    return {
        "arm": arm,
        "instruction": INSTRUCTION,
        "question_ID": public.question_id,
        "evidence_sha256": public.evidence_sha256,
        "history": visible["history"] if direct else None,
        "step_directory": None
        if direct
        else [
            {key: step[key] for key in ("step", "name", "role")}
            for step in visible["history"]
        ],
        "tools": []
        if direct
        else ["read_step", "causal_walk"]
        if arm == "vestige"
        else ["read_step"],
        "evidence_policy": "all original trace records available; no gold or cross-arm memory",
        "equivalence_group": "stateless_full_history" if direct else arm,
        "public_payload_sha256": public.evidence_sha256,
    }


def read_trace_step(public: PublicTrajectory, step: int) -> dict:
    """Dynamic direct-record control; no search, persistence, or causal claims."""
    public = validate_public(public.as_dict())
    if type(step) is not int or not 0 <= step < len(public.history):
        raise ValueError("invalid exact step handle")
    return public.as_dict()["history"][step]

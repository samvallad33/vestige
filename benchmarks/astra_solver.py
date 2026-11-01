"""Stateless Astra (GPT-6 Astra) solver factory for Terminal-Bench via Vestige."""

import copy
import os
import sys
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "upstream" / "agentdojo-library"
if str(LIBRARY) not in sys.path:
    sys.path.insert(0, str(LIBRARY))
if str(LIBRARY / "src") not in sys.path:
    sys.path.insert(0, str(LIBRARY / "src"))

import openai
from agentdojo.functions_runtime import EmptyEnv
from vestige_eval.guarded_pipeline import AuditedSolver
from vestige_eval.responses_model import ResponsesLLM
from vestige_eval.stack import NativeInterdiction, canonical, digest

PINNED_ASTRA_MODEL = "gpt-6-astra"


class AuditedAstraSolver(AuditedSolver):
    """Audited solver for OpenAI Responses API models (GPT-6 Astra).

    Persists exact raw provider events to provider.jsonl.
    For Operator Gate screening before sending to the model, masks base64-encoded
    reasoning ciphertext in prior assistant messages to prevent high-entropy
    ciphertext substrings from triggering secret/credential leak detectors (OP-036).
    The raw request/response is retained with 100% fidelity on disk.
    """

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args=None):
        if not self.lock.acquire(blocking=False):
            raise NativeInterdiction("concurrent use of one audited solver")
        extra = dict(extra_args or {})
        context = extra.pop("_vestige_native_context", None)
        session = extra.get("session_id") or "control-" + uuid.uuid4().hex
        folder = self.evidence_root / session
        try:
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        except BaseException:
            self.lock.release()
            raise
        pending = {}
        previous = self.solver.audit

        def persist(event, data):
            with (folder / "provider.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical({"event": event, **data}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        def audit(event, data):
            persist(event, data)
            call = "provider-" + data["request_id"]
            if event == "responses_request" and context:
                stack, sid = context
                gate_req = copy.deepcopy(data["request"])
                for item in gate_req.get("input", []):
                    if item.get("type") == "reasoning" and "encrypted_content" in item:
                        item["encrypted_content"] = "[encrypted_reasoning_token]"
                args = {"request": gate_req}
                admitted = stack.request("before", session=sid, call=call, tool="openai_responses", args=args)
                if not admitted.get("allowed"):
                    raise NativeInterdiction("provider request BLOCKED_BY_OPERATOR")
                pending.update(call=call, args=args)
            elif event == "responses_response" and context:
                stack, sid = context
                admitted_args = pending["args"]
                pending.clear()
                stack.request(
                    "after",
                    session=sid,
                    call=call,
                    tool="openai_responses",
                    args=admitted_args,
                    result=data["response"],
                )
                stack.record(
                    sid, {"kind": "provider_response", "call": call, "response_sha256": digest(data["response"])}
                )
            if previous:
                previous(event, data)

        self.solver.audit = audit
        try:
            result = self.solver.query(query, runtime, env, messages, extra)
            if pending:
                raise NativeInterdiction("provider returned without acknowledged raw response")
            if context:
                result[-1]["_vestige_native_context"] = context
            return result
        except BaseException as exc:
            persist(
                "solver_interdiction",
                {"error_type": type(exc).__name__, "raw_response": getattr(exc, "raw_response", None)},
            )
            if pending and context:
                stack, sid = context
                stack.request(
                    "after",
                    session=sid,
                    call=pending["call"],
                    tool="openai_responses",
                    args=pending["args"],
                    result=None,
                    error=type(exc).__name__,
                )
            raise
        finally:
            self.solver.audit = previous
            self.lock.release()


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        env_file = ROOT / ".env.openai"
        if env_file.exists():
            key = env_file.read_text().strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env.openai")
    return key


def factory(logs_dir: Path, model_name: str = PINNED_ASTRA_MODEL) -> AuditedAstraSolver:
    key = get_api_key()
    client = openai.OpenAI(api_key=key)
    # Calibrated reasoning effort: high
    llm = ResponsesLLM(
        client=client,
        model=model_name or PINNED_ASTRA_MODEL,
        reasoning_effort="high",
        max_output_tokens=32000,
        max_requests=40,
        request_timeout_seconds=300,
    )
    return AuditedAstraSolver(llm, logs_dir)


def validate_astra_solver(solver: AuditedSolver, model_name: str = None) -> dict:
    if not isinstance(solver, AuditedSolver):
        raise TypeError("Astra solver must be wrapped in AuditedSolver")
    if not isinstance(solver.solver, ResponsesLLM):
        raise TypeError("Audited solver must wrap ResponsesLLM")
    effective_model = model_name or PINNED_ASTRA_MODEL
    if solver.solver.model != effective_model:
        raise ValueError(f"model mismatch: {solver.solver.model} != {effective_model}")
    return {
        "mode": "frontier_responses",
        "model": effective_model,
        "provider": "openai",
        "reasoning_effort": solver.solver.reasoning_effort,
    }

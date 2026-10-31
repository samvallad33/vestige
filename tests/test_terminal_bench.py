"""Terminal seam and immutable-plan verification, no provider or Docker needed."""

import asyncio
import json

import pytest
from harbor.environments.base import ExecResult

from campaigns.terminal_bench.adapter import (
    EmptyEnv,
    GuardedRuntime,
    VestigeTerminalAgent,
    terminal_runtime,
    validate_terminal_solver,
)
from campaigns.terminal_bench.prepare import check, prepare


class FakeEnvironment:
    def __init__(self):
        self.calls = []

    async def exec(self, **kwargs):
        self.calls.append(kwargs)
        return ExecResult(stdout="marker", stderr="", return_code=0)


class FakeStack:
    catalog = ()

    def __init__(self, allowed=True, error=None):
        self.allowed, self.error, self.events = allowed, error, []

    def request(self, method, **kwargs):
        self.events.append((method, kwargs))
        if self.error:
            raise self.error
        return {"allowed": self.allowed, "reason": "BLOCKED_BY_OPERATOR"}

    def record(self, session, data):
        self.events.append(("record", data))


def drive(stack):
    async def main():
        env = FakeEnvironment()
        runtime = GuardedRuntime(
            terminal_runtime(env, asyncio.get_running_loop()), stack, "session"
        )
        runtime.model_call = "model-call"
        result = await asyncio.to_thread(
            runtime.run_function,
            EmptyEnv(),
            "Bash",
            {"command": "printf marker", "timeout_sec": 10},
        )
        return env, result

    return asyncio.run(main())


def test_terminal_admission_wraps_exact_harbor_exec_payload():
    stack = FakeStack()
    env, (result, error) = drive(stack)
    assert error is None and result["stdout"] == "marker"
    assert env.calls == [{"command": "printf marker", "timeout_sec": 10}]
    assert [event for event, _ in stack.events] == ["before", "after", "record"]
    assert stack.events[0][1]["tool"] == "Bash"
    assert stack.events[0][1]["args"] == {"command": "printf marker", "timeout_sec": 10}


def test_denial_prevents_harbor_execution():
    stack = FakeStack(allowed=False)
    env, (_, error) = drive(stack)
    assert error == "BLOCKED_BY_OPERATOR" and env.calls == []
    assert [event for event, _ in stack.events] == ["before"]


def test_transport_fault_prevents_harbor_execution():
    stack = FakeStack(error=RuntimeError("FAIL_CLOSED_INTERDICTION"))
    with pytest.raises(RuntimeError, match="FAIL_CLOSED"):
        drive(stack)
    assert [event for event, _ in stack.events] == ["before"]


def test_missing_solver_is_launch_gated_before_commands(tmp_path):
    agent = VestigeTerminalAgent(logs_dir=tmp_path)
    env = FakeEnvironment()
    with pytest.raises(RuntimeError, match="LAUNCH_GATED"):
        asyncio.run(agent.setup(env))
    assert env.calls == []


def test_unbudgeted_responses_cannot_reach_transport_or_environment(tmp_path):
    from vestige_eval.guarded_pipeline import AuditedSolver
    from vestige_eval.responses_model import ResponsesLLM

    class Transport:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise AssertionError("transport must remain untouched")

    transport = Transport()
    client = type("Client", (), {"responses": transport})()
    solver = AuditedSolver(ResponsesLLM(client, budget=None), tmp_path / "provider")
    solver.synthetic_zero_inference = (
        True  # A boolean cannot bypass exact type/factory admission.
    )
    environment = FakeEnvironment()
    with pytest.raises(RuntimeError, match="LAUNCH_GATED"):
        validate_terminal_solver(solver, "unbudgeted:factory")
    assert transport.calls == 0 and environment.calls == []


def test_unapproved_factory_rejected_before_construction(tmp_path):
    agent = VestigeTerminalAgent(logs_dir=tmp_path, solver_factory="unbudgeted:factory")
    environment = FakeEnvironment()
    with pytest.raises(RuntimeError, match="unsupported factory"):
        asyncio.run(agent.setup(environment))
    assert environment.calls == []


def test_exact_free_local_contract_validates_without_inference(tmp_path, monkeypatch):
    from local_solver import (
        PINNED_MODEL,
        LocalAuditedSolver,
        LocalBounds,
        LocalBudget,
        LoopbackTransport,
        OllamaLLM,
    )

    requests = []

    def forbidden(*args, **kwargs):
        requests.append(args)
        raise AssertionError("validation must not request metadata or inference")

    monkeypatch.setattr(LoopbackTransport, "exchange", forbidden)
    budget = LocalBudget(LocalBounds())
    solver = LocalAuditedSolver(
        OllamaLLM(LoopbackTransport(), budget), tmp_path / "provider"
    )
    solver.model_metadata = {"name": PINNED_MODEL, "digest": budget.bounds.model_digest}
    contract = validate_terminal_solver(solver, "local_solver:factory", PINNED_MODEL)
    assert (
        contract["provider"] == "ollama_chat" and contract["provider_cost_usd"] == "0"
    )
    assert requests == [] and budget.records == []
    agent = VestigeTerminalAgent(
        logs_dir=tmp_path,
        model_name=PINNED_MODEL,
        solver_factory="local_solver:factory",
        skip_model_parsing=True,
    )
    asyncio.run(agent.setup(FakeEnvironment()))
    assert requests == []


def test_free_local_fee_tamper_rejected_without_inference(tmp_path):
    from local_solver import (
        PINNED_MODEL,
        LocalAuditedSolver,
        LocalBounds,
        LocalBudget,
        LocalSolverError,
        LoopbackTransport,
        OllamaLLM,
    )

    budget = LocalBudget(LocalBounds())
    solver = LocalAuditedSolver(
        OllamaLLM(LoopbackTransport(), budget), tmp_path / "provider"
    )
    solver.model_metadata = {"name": PINNED_MODEL, "digest": budget.bounds.model_digest}
    budget.provider_cost_usd = "1"
    with pytest.raises(LocalSolverError, match="identity/accounting contract changed"):
        validate_terminal_solver(solver, "local_solver:factory", PINNED_MODEL)
    assert budget.records == []


def test_frozen_plan_checks_sources_and_detects_tampering(tmp_path):
    path = tmp_path / "plan.json"
    plan = prepare(path)
    assert plan["dataset"] == "terminal-bench/terminal-bench@4.0.0"
    assert plan["cpu_candidates"] and plan["provider_calls"] == 0
    assert check(path)["status"] == "PASS"
    plan["cpu_candidates"] = []
    path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        check(path)


def test_freeze_never_overwrites_prior_evidence(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare(path)
    assert path.read_text() == "preserve"


def test_same_head_dirty_tracked_source_is_rejected(tmp_path, monkeypatch):
    from campaigns.terminal_bench import prepare as preparation

    path = tmp_path / "plan.json"
    plan = preparation.prepare(path)
    assert (
        "vestige_eval/guarded_pipeline.py"
        in plan["sources"]["agentdojo-library"]["execution_sha256"]
    )
    original = preparation.git

    def dirty(repo, *args):
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return " M execution.py"
        return original(repo, *args)

    monkeypatch.setattr(preparation, "git", dirty)
    with pytest.raises(RuntimeError, match="tracked source is dirty"):
        preparation.check(path)


def test_execution_bytes_checked_independently_of_head(tmp_path, monkeypatch):
    from campaigns.terminal_bench import prepare as preparation

    path = tmp_path / "plan.json"
    preparation.prepare(path)
    original = preparation.sha

    def changed(source):
        if str(source).endswith("vestige_eval/guarded_pipeline.py"):
            return "0" * 64
        return original(source)

    monkeypatch.setattr(preparation, "sha", changed)
    with pytest.raises(RuntimeError, match="execution source bytes drift"):
        preparation.check(path)
